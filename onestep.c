#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
  NL          = 8,       /* book levels (visible) */
  TMAX        = 500,     /* max entries in a KV table */
  SP_MAX      = 64,      /* spread upper bound for per-sp tables */
  IMB_BINS    = 6,       /* imb_bin cardinality */
  N_VIS       = 5,       /* visible pooled event types: tp, tm_q, tm_c, dp, dm */
  N_HAWKES    = 6,       /* + hp (hidden-book surfacings) */
  REC_COLS    = 54,      /* int32 columns per event record (train.events) */
  ROW_COLS    = 49,      /* int32 columns per raw book row */
  EV_IDLE     = 8,       /* event-type marker for end-of-row (train.events) */
  TICK        = 2        /* even-tick quantization step */
};

enum {                   /* pooled event type indices (ask/bid merged) */
  EV_TP, EV_TM_Q, EV_TM_C, EV_DP, EV_DM, EV_HP
};

#define HK_BETA           0.05     /* fixed Hawkes kernel decay */
#define DIFFUSE_THRESHOLD 0.95     /* refill.k=2 mass below this ⇒ sample tail */
#define TAIL_ALPHA_MAX   -1.05     /* need α < this for finite tail integral */
#define TAIL_CAP_FACTOR   3        /* truncate power law at k_max = F·k_cutoff */
#define U_FLOOR           1e-12    /* prevent log(0) in Ogata sampling */
#define MU_SCALE_MIN      0.01     /* baseline-μ split clamp */
#define MU_SCALE_MAX      1.00
#define SP_LIMIT_MULT     10       /* bad-state guard: sp > sp0·MULT+OFFSET */
#define SP_LIMIT_OFFSET   100

struct Row {
  int32_t aR[NL], bR[NL], aS[NL], bS[NL], aN[NL], bN[NL], y;
  int32_t aF[NL], bF[NL];        /* F: 1 = refill-placeholder level */
};

struct Side {                    /* view into r for one side (ask or bid) */
  int32_t *R, *N, *S, *F;
};

struct KV { int n; double k[TMAX]; double v[TMAX]; };

struct TailParam { double alpha; int k_cutoff; double f_tail; };

static int is_events_fmt = 0;
static FILE *input_fp = NULL;
static off_t bytes_remaining = -1;
static const char *events_path = NULL, *idx_path = NULL;
static long session_id = -1;

static struct KV tp_a[IMB_BINS], tp_b[IMB_BINS];
static struct KV tm_a[IMB_BINS], tm_b[IMB_BINS];
static struct KV dp_a[IMB_BINS], dp_b[IMB_BINS];
static struct KV dm_a[IMB_BINS], dm_b[IMB_BINS];
static struct KV n_imb[IMB_BINS];
static int imb_loaded = 0;

static struct KV tp_own, dp_own;
static struct KV tp_own_sp[SP_MAX], dp_own_sp[SP_MAX];

static struct KV refill;              /* pooled ask+bid refill histogram */
static int refill_diffuse = 0;

static struct TailParam tail = {0.0, 0, 0.0};   /* pooled tail params */
static int use_tail = 0;

static int hk_loaded = 0;
static int use_state_mu = 0;
static double hk_mu[N_HAWKES], hk_alpha[N_HAWKES][N_HAWKES];
static double hk_phi[N_HAWKES], hk_phi_stat[N_HAWKES];
static double hk_mu_scale = 1.0;          /* shared scale for state-μ baseline */

static int reset_phi = 1;

/* ── side dispatch ────────────────────────────────────────────────────────── */

static struct Side side_view(struct Row *r, int side) {
  struct Side s;
  s.R = side ? r->bR : r->aR;
  s.N = side ? r->bN : r->aN;
  s.S = side ? r->bS : r->aS;
  s.F = side ? r->bF : r->aF;
  return s;
}

static int imb_bin(int32_t aN0, int32_t bN0, int32_t aN1, int32_t bN1) {
  int64_t s = (int64_t)aN0 + bN0, d = (int64_t)aN0 - bN0;
  int b0 = (s == 0) ? 1 : (d * 5 < -s) ? 0 : (d * 5 > s) ? 2 : 1;
  return b0 * 2 + ((aN1 > bN1) ? 1 : 0);
}

/* ── I/O ──────────────────────────────────────────────────────────────────── */

static int read_wire(struct Row *r, FILE *f) {
  int32_t w[REC_COLS];
  if (is_events_fmt) {
    const off_t recsz = (off_t)sizeof w;
    while (1) {
      if (bytes_remaining >= 0 && bytes_remaining < recsz) return 0;
      if (fread(w, recsz, 1, f) != 1) return 0;
      if (bytes_remaining >= 0) bytes_remaining -= recsz;
      if (w[0] == EV_IDLE) break;
    }
    memset(r, 0, sizeof *r);
    memcpy(r->aR, &w[5 + 0 * NL], NL * 4);
    memcpy(r->bR, &w[5 + 1 * NL], NL * 4);
    memcpy(r->aS, &w[5 + 2 * NL], NL * 4);
    memcpy(r->bS, &w[5 + 3 * NL], NL * 4);
    memcpy(r->aN, &w[5 + 4 * NL], NL * 4);
    memcpy(r->bN, &w[5 + 5 * NL], NL * 4);
    r->y = w[5 + 6 * NL];
    return 1;
  }
  if (fread(w, ROW_COLS * sizeof(int32_t), 1, f) != 1) return 0;
  memset(r, 0, sizeof *r);
  memcpy(r->aR, &w[0 * NL], NL * 4);
  memcpy(r->bR, &w[1 * NL], NL * 4);
  memcpy(r->aS, &w[2 * NL], NL * 4);
  memcpy(r->bS, &w[3 * NL], NL * 4);
  memcpy(r->aN, &w[4 * NL], NL * 4);
  memcpy(r->bN, &w[5 * NL], NL * 4);
  r->y = w[6 * NL];
  return 1;
}

static int write_wire(struct Row *r, FILE *f) {
  int32_t w[ROW_COLS];
  memcpy(&w[0 * NL], r->aR, NL * 4);
  memcpy(&w[1 * NL], r->bR, NL * 4);
  memcpy(&w[2 * NL], r->aS, NL * 4);
  memcpy(&w[3 * NL], r->bS, NL * 4);
  memcpy(&w[4 * NL], r->aN, NL * 4);
  memcpy(&w[5 * NL], r->bN, NL * 4);
  w[6 * NL] = r->y;
  return fwrite(w, sizeof w, 1, f) == 1;
}

/* ── KV tables ────────────────────────────────────────────────────────────── */

static int load_kv(const char *path, struct KV *t) {
  t->n = 0;
  FILE *f = fopen(path, "r");
  if (!f) return 0;
  while (t->n < TMAX && fscanf(f, "%lf %lf", &t->k[t->n], &t->v[t->n]) == 2) t->n++;
  fclose(f);
  return t->n;
}

static double lookup(struct KV *t, double k) {
  if (t->n == 0) return 0.0;
  if (k <= t->k[0]) return t->v[0];
  if (k >= t->k[t->n - 1]) return t->v[t->n - 1];
  for (int i = 1; i < t->n; i++)
    if (k <= t->k[i]) {
      double a = (k - t->k[i - 1]) / (t->k[i] - t->k[i - 1]);
      return t->v[i - 1] * (1 - a) + t->v[i] * a;
    }
  return t->v[t->n - 1];
}

static double sample_dist(struct KV *t) {
  double total = 0, cum = 0;
  for (int i = 0; i < t->n; i++) total += t->v[i];
  if (total <= 0) return 0;
  double u = drand48() * total;
  for (int i = 0; i < t->n; i++) {
    cum += t->v[i];
    if (u <= cum) return t->k[i];
  }
  return t->k[t->n - 1];
}

static struct KV *pick_sp_kv(struct KV *sp_tbl, int32_t sp, struct KV *fallback) {
  if (sp >= 0 && sp < SP_MAX && sp_tbl[sp].n > 0) return &sp_tbl[sp];
  return fallback;
}

/* ── refill sampler (empirical + optional power-law tail) ─────────────────── */

static void compute_diffuse_flag(struct KV *t, int *diffuse) {
  double total = 0, mass_at_2 = 0;
  for (int i = 0; i < t->n; i++) {
    total += t->v[i];
    if ((int)t->k[i] == TICK) mass_at_2 = t->v[i];
  }
  *diffuse = (total > 0 && mass_at_2 / total < DIFFUSE_THRESHOLD);
}

static void load_tail_param(const char *path, struct TailParam *t) {
  t->alpha = 0.0; t->k_cutoff = 0; t->f_tail = 0.0;
  FILE *f = fopen(path, "r");
  if (!f) return;
  char tag[32]; double v; int iv;
  while (fscanf(f, "%31s", tag) == 1) {
    if (tag[0] == '#') { int c; while ((c = fgetc(f)) != '\n' && c != EOF) { } continue; }
    if      (!strcmp(tag, "alpha")    && fscanf(f, "%lf", &v)  == 1) t->alpha    = v;
    else if (!strcmp(tag, "k_cutoff") && fscanf(f, "%d",  &iv) == 1) t->k_cutoff = iv;
    else if (!strcmp(tag, "f_tail")   && fscanf(f, "%lf", &v)  == 1) t->f_tail   = v;
    else if (!strcmp(tag, "max_obs")  && fscanf(f, "%d",  &iv) == 1) (void)iv;
    else    { if (fscanf(f, "%lf", &v) != 1) break; }
  }
  fclose(f);
}

static int32_t sample_tail(const struct TailParam *t) {
  if (t->alpha >= TAIL_ALPHA_MAX || t->k_cutoff <= 0) return (int32_t)t->k_cutoff;
  int32_t k_max = TAIL_CAP_FACTOR * t->k_cutoff;
  double U = drand48(); if (U < U_FLOOR) U = U_FLOOR;
  double ap = pow((double)t->k_cutoff, t->alpha + 1.0);
  double bp = pow((double)k_max,       t->alpha + 1.0);
  double k  = pow(U * (bp - ap) + ap,  1.0 / (t->alpha + 1.0));
  int32_t ki = (int32_t)(k + 0.5);
  if (ki < t->k_cutoff) ki = t->k_cutoff;
  if (ki > k_max)       ki = k_max;
  if (ki & 1)           ki++;
  return ki;
}

static int32_t sample_refill(void) {
  if (use_tail && tail.f_tail > 0.0 && drand48() < tail.f_tail)
    return sample_tail(&tail);
  int32_t dist = (refill_diffuse && refill.n > 0) ? (int32_t)sample_dist(&refill) : TICK;
  return dist <= 0 ? TICK : dist;
}

/* ── Hawkes params (10-D) ─────────────────────────────────────────────────── */

static int load_hawkes(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) return 0;
  for (int c = 0; c < N_HAWKES; c++) {
    hk_mu[c] = 0;
    for (int j = 0; j < N_HAWKES; j++) hk_alpha[c][j] = 0;
  }
  char tag[32]; double v; int c, j;
  while (fscanf(f, "%31s", tag) == 1) {
    if (!strcmp(tag, "beta")) { double b; if (fscanf(f, "%lf", &b) != 1) break; }
    else if (!strcmp(tag, "mu")) {
      if (fscanf(f, "%d %lf", &c, &v) != 2) break;
      if (c >= 0 && c < N_HAWKES) hk_mu[c] = v;
    } else if (!strcmp(tag, "alpha")) {
      if (fscanf(f, "%d %d %lf", &c, &j, &v) != 3) break;
      if (c >= 0 && c < N_HAWKES && j >= 0 && j < N_HAWKES) hk_alpha[c][j] = v;
    } else break;
  }
  fclose(f);

  /* Stationary λ via Gauss elimination of (I − α/β)·λ = μ. */
  double A[N_HAWKES][N_HAWKES + 1];
  for (int c = 0; c < N_HAWKES; c++) {
    for (int j = 0; j < N_HAWKES; j++)
      A[c][j] = (c == j ? 1.0 : 0.0) - hk_alpha[c][j] / HK_BETA;
    A[c][N_HAWKES] = hk_mu[c];
  }
  for (int p = 0; p < N_HAWKES; p++) {
    int piv = p; double best = fabs(A[p][p]);
    for (int r = p + 1; r < N_HAWKES; r++)
      if (fabs(A[r][p]) > best) { best = fabs(A[r][p]); piv = r; }
    if (piv != p)
      for (int cc = 0; cc < N_HAWKES + 1; cc++) {
        double t = A[p][cc]; A[p][cc] = A[piv][cc]; A[piv][cc] = t;
      }
    for (int r = 0; r < N_HAWKES; r++)
      if (r != p && A[r][p] != 0.0) {
        double f = A[r][p] / A[p][p];
        for (int cc = p; cc < N_HAWKES + 1; cc++) A[r][cc] -= f * A[p][cc];
      }
  }
  for (int jj = 0; jj < N_HAWKES; jj++) {
    hk_phi_stat[jj] = (A[jj][N_HAWKES] / A[jj][jj]) / HK_BETA;
    hk_phi[jj]      = hk_phi_stat[jj];
  }
  double total_mu = 0, total_stat = 0;
  for (int jj = 0; jj < N_HAWKES; jj++) {
    total_mu   += hk_mu[jj];
    total_stat += hk_phi_stat[jj] * HK_BETA;
  }
  double scale = (total_stat > 0) ? total_mu / total_stat : 1.0;
  if (scale < MU_SCALE_MIN) scale = MU_SCALE_MIN;
  if (scale > MU_SCALE_MAX) scale = MU_SCALE_MAX;
  hk_mu_scale = scale;
  return 1;
}

/* ── book event application ───────────────────────────────────────────────── */

static void shift_down_from(struct Side s, int from) {
  for (int k = NL - 1; k > from; k--) {
    s.R[k] = s.R[k - 1]; s.N[k] = s.N[k - 1];
    s.S[k] = s.S[k - 1]; s.F[k] = s.F[k - 1];
  }
}

static void shift_up_from(struct Side s, int from) {
  for (int k = from; k < NL - 1; k++) {
    s.R[k] = s.R[k + 1]; s.N[k] = s.N[k + 1];
    s.S[k] = s.S[k + 1]; s.F[k] = s.F[k + 1];
  }
}

static void cascade_refill(struct Row *r, int side) {
  struct Side s = side_view(r, side);
  if (s.N[NL - 2] > 0) {
    int32_t dist = sample_refill();
    s.R[NL - 1] = side ? s.R[NL - 2] - dist : s.R[NL - 2] + dist;
    s.N[NL - 1] = 1; s.S[NL - 1] = 1; s.F[NL - 1] = 1;
  } else {
    s.R[NL - 1] = 0; s.N[NL - 1] = 0; s.S[NL - 1] = 0; s.F[NL - 1] = 0;
  }
}

static void apply_tp(struct Row *r, int side, int32_t dist) {
  struct Side s = side_view(r, side);
  if (dist <= 0) { s.N[0]++; s.S[0]++; return; }
  int32_t opp = side ? r->aR[0] : r->bR[0];
  int32_t newR = side ? s.R[0] + dist : s.R[0] - dist;
  int crosses = side ? (newR >= opp) : (newR <= opp);
  if (crosses) {
    newR = side ? opp - TICK : opp + TICK;
    crosses = side ? (newR >= opp) : (newR <= opp);
    if (crosses) { s.N[0]++; s.S[0]++; return; }
  }
  shift_down_from(s, 0);
  s.R[0] = newR; s.N[0] = 1; s.S[0] = 1; s.F[0] = 0;
}

static void apply_dp(struct Row *r, int side, int32_t dist) {
  struct Side s = side_view(r, side);
  if (dist <= 0) return;
  int32_t newR = side ? s.R[0] - dist : s.R[0] + dist;
  int k;
  for (k = 1; k < NL; k++) {
    if (s.N[k] == 0) break;
    if (s.R[k] == newR) {
      if (s.F[k] == 0) { s.N[k]++; s.S[k]++; }
      else             { s.N[k] = 1; s.S[k] = 1; s.F[k] = 0; }
      return;
    }
    int past = side ? (newR > s.R[k]) : (newR < s.R[k]);
    if (past) break;
  }
  if (k == NL) return;
  shift_down_from(s, k);
  s.R[k] = newR; s.N[k] = 1; s.S[k] = 1; s.F[k] = 0;
}

static void apply_tm(struct Row *r, int side) {
  struct Side s = side_view(r, side);
  if (s.N[0] == 0) return;
  s.N[0]--; if (s.S[0] > 0) s.S[0]--;
  if (s.N[0] == 0) { shift_up_from(s, 0); cascade_refill(r, side); }
}

static void apply_dm(struct Row *r, int side) {
  struct Side s = side_view(r, side);
  int total = 0;
  for (int k = 1; k < NL; k++) if (s.N[k] > 0 && s.F[k] == 0) total += s.N[k];
  if (total == 0) return;
  int u = (int)(drand48() * total), sum = 0, pick = 1;
  for (int k = 1; k < NL; k++) {
    if (s.N[k] == 0 || s.F[k] != 0) continue;
    sum += s.N[k];
    if (u < sum) { pick = k; break; }
  }
  s.N[pick]--; if (s.S[pick] > 0) s.S[pick]--;
  if (s.N[pick] == 0) { shift_up_from(s, pick); cascade_refill(r, side); }
}

/* ── rate computation ─────────────────────────────────────────────────────── */

/* Pool side-specific imb rates: return (ask+bid) rate for given pooled type.
 * tm is split: tm_q and tm_c share the tm imb-table total (state-gated later). */
static double pool_rate(int type, int im, int32_t sp) {
  switch (type) {
    case EV_TP:   return lookup(&tp_a[im], sp) + lookup(&tp_b[im], sp);
    case EV_TM_Q: return lookup(&tm_a[im], sp) + lookup(&tm_b[im], sp);
    case EV_TM_C: return lookup(&tm_a[im], sp) + lookup(&tm_b[im], sp);
    case EV_DP:   return lookup(&dp_a[im], sp) + lookup(&dp_b[im], sp);
    case EV_DM:   return lookup(&dm_a[im], sp) + lookup(&dm_b[im], sp);
    default:      return 0;
  }
}

/* tm_q requires a side with N[0]>1; tm_c requires a side with N[0]=1. */
static int tm_q_available(struct Row *r) { return r->aN[0] > 1 || r->bN[0] > 1; }
static int tm_c_available(struct Row *r) { return r->aN[0] == 1 || r->bN[0] == 1; }

static double compute_rates(struct Row *r, double rates[N_HAWKES]) {
  int32_t sp = r->aR[0] - r->bR[0];
  double total = 0;
  if (hk_loaded) {
    double mu[N_HAWKES];
    if (use_state_mu && imb_loaded) {
      int im = imb_bin(r->aN[0], r->bN[0], r->aN[1], r->bN[1]);
      for (int k = 0; k < N_VIS; k++) mu[k] = pool_rate(k, im, sp) * hk_mu_scale;
      mu[EV_HP] = hk_mu[EV_HP];
    } else {
      for (int k = 0; k < N_HAWKES; k++) mu[k] = hk_mu[k];
    }
    for (int k = 0; k < N_HAWKES; k++) {
      double rc = mu[k];
      for (int j = 0; j < N_HAWKES; j++) rc += hk_alpha[k][j] * hk_phi[j];
      rates[k] = rc > 0 ? rc : 0;
    }
    /* State-gate tm_q / tm_c to available book configurations. */
    if (!tm_q_available(r)) rates[EV_TM_Q] = 0;
    if (!tm_c_available(r)) rates[EV_TM_C] = 0;
    for (int k = 0; k < N_HAWKES; k++) total += rates[k];
    return total;
  }
  /* Fallback: pooled imb rates, no Hawkes excitation. */
  if (imb_loaded) {
    int im = imb_bin(r->aN[0], r->bN[0], r->aN[1], r->bN[1]);
    for (int k = 0; k < N_VIS; k++) rates[k] = pool_rate(k, im, sp);
    if (!tm_q_available(r)) rates[EV_TM_Q] = 0;
    if (!tm_c_available(r)) rates[EV_TM_C] = 0;
    rates[EV_HP] = 0;
    for (int k = 0; k < N_VIS; k++) total += rates[k];
    if (total <= 0) return 0;
    double nr = lookup(&n_imb[im], sp);
    if (nr > 0 && nr < 1) {
      double scale = (1.0 - nr) / total;
      for (int k = 0; k < N_VIS; k++) rates[k] *= scale;
      total = 1.0 - nr;
    }
  }
  return total;
}

/* After picking pooled type, choose side by ratio of side-conditional imb
 * rates at current state, restricted to sides compatible with the type.
 * tm_q needs N[0]>1; tm_c needs N[0]==1. */
static int sample_side(int type, struct Row *r) {
  int allow_a = 1, allow_b = 1;
  if (type == EV_TM_Q) { allow_a = r->aN[0] > 1;  allow_b = r->bN[0] > 1;  }
  if (type == EV_TM_C) { allow_a = r->aN[0] == 1; allow_b = r->bN[0] == 1; }
  if (allow_a && !allow_b) return 0;
  if (!allow_a && allow_b) return 1;
  if (!allow_a && !allow_b) return 0;
  if (!imb_loaded) return drand48() < 0.5 ? 0 : 1;
  int32_t sp = r->aR[0] - r->bR[0];
  int im = imb_bin(r->aN[0], r->bN[0], r->aN[1], r->bN[1]);
  struct KV *a, *b;
  switch (type) {
    case EV_TP:                                 a = &tp_a[im]; b = &tp_b[im]; break;
    case EV_TM_Q: case EV_TM_C:                 a = &tm_a[im]; b = &tm_b[im]; break;
    case EV_DP:                                 a = &dp_a[im]; b = &dp_b[im]; break;
    case EV_DM:                                 a = &dm_a[im]; b = &dm_b[im]; break;
    default:    return drand48() < 0.5 ? 0 : 1;
  }
  double ra = lookup(a, sp), rb = lookup(b, sp);
  if (ra + rb <= 0) return drand48() < 0.5 ? 0 : 1;
  return drand48() * (ra + rb) < ra ? 0 : 1;
}

/* ── Ogata-thinned simulation loop ────────────────────────────────────────── */

static int bad_state(struct Row *r, int32_t sp_limit) {
  int32_t sp = r->aR[0] - r->bR[0];
  if (r->aN[0] == 0 || r->bN[0] == 0) { r->y = 1; return 1; }
  if (sp > sp_limit || sp <= 0)       { r->y = 2; return 1; }
  return 0;
}

static void fire_event(struct Row *r, int type, int side, int32_t sp) {
  switch (type) {
    case EV_TP:   apply_tp(r, side, (int32_t)sample_dist(pick_sp_kv(tp_own_sp, sp, &tp_own))); break;
    case EV_TM_Q: apply_tm(r, side); break;                  /* N[0]>1 → queue decrement */
    case EV_TM_C: apply_tm(r, side); break;                  /* N[0]=1 → cascade */
    case EV_DP:   apply_dp(r, side, (int32_t)sample_dist(pick_sp_kv(dp_own_sp, sp, &dp_own))); break;
    case EV_DM:   apply_dm(r, side); break;
  }
}

static void simulate(struct Row *r, double T) {
  double t = 0;
  int32_t sp0 = r->aR[0] - r->bR[0];
  int32_t sp_limit = (sp0 > 0 ? sp0 : TICK) * SP_LIMIT_MULT + SP_LIMIT_OFFSET;
  r->y = 0;
  if (reset_phi && hk_loaded)
    for (int k = 0; k < N_HAWKES; k++) hk_phi[k] = hk_phi_stat[k];
  int n_rates = hk_loaded ? N_HAWKES : N_VIS;

  while (t < T) {
    if (bad_state(r, sp_limit)) break;
    double rates[N_HAWKES];
    double lam_star = compute_rates(r, rates);
    if (lam_star <= 0) break;
    double dt = -log(drand48()) / lam_star;
    if (t + dt > T) break;
    if (hk_loaded) {
      double f = exp(-HK_BETA * dt);
      for (int k = 0; k < N_HAWKES; k++) hk_phi[k] *= f;
    }
    t += dt;
    double lam_now = compute_rates(r, rates);
    if (drand48() * lam_star >= lam_now) continue;            /* Ogata reject */
    double u = drand48() * lam_now, cum = 0;
    int pick = n_rates - 1;
    for (int k = 0; k < n_rates; k++) {
      cum += rates[k];
      if (u < cum) { pick = k; break; }
    }
    if (hk_loaded) hk_phi[pick] += 1.0;
    if (pick == EV_HP) continue;                              /* HP phantom: φ only */
    int side = sample_side(pick, r);
    fire_event(r, pick, side, r->aR[0] - r->bR[0]);
    if (bad_state(r, sp_limit)) return;
  }
}

/* ── table loading orchestrator ───────────────────────────────────────────── */

static void load_tables(const char *dir) {
  char path[512];
  imb_loaded = 1;
  for (int im = 0; im < IMB_BINS; im++) {
    snprintf(path, sizeof path, "%s/tp.a.imb%d.rates", dir, im); if (!load_kv(path, &tp_a[im])) imb_loaded = 0;
    snprintf(path, sizeof path, "%s/tp.b.imb%d.rates", dir, im); if (!load_kv(path, &tp_b[im])) imb_loaded = 0;
    snprintf(path, sizeof path, "%s/tm.a.imb%d.rates", dir, im); if (!load_kv(path, &tm_a[im])) imb_loaded = 0;
    snprintf(path, sizeof path, "%s/tm.b.imb%d.rates", dir, im); if (!load_kv(path, &tm_b[im])) imb_loaded = 0;
    snprintf(path, sizeof path, "%s/dp.a.imb%d.rates", dir, im); if (!load_kv(path, &dp_a[im])) imb_loaded = 0;
    snprintf(path, sizeof path, "%s/dp.b.imb%d.rates", dir, im); if (!load_kv(path, &dp_b[im])) imb_loaded = 0;
    snprintf(path, sizeof path, "%s/dm.a.imb%d.rates", dir, im); if (!load_kv(path, &dm_a[im])) imb_loaded = 0;
    snprintf(path, sizeof path, "%s/dm.b.imb%d.rates", dir, im); if (!load_kv(path, &dm_b[im])) imb_loaded = 0;
    snprintf(path, sizeof path, "%s/n.imb%d.rates",    dir, im); load_kv(path, &n_imb[im]);
  }
  snprintf(path, sizeof path, "%s/tp.own", dir); load_kv(path, &tp_own);
  snprintf(path, sizeof path, "%s/dp.own", dir); load_kv(path, &dp_own);
  for (int sp = 0; sp < SP_MAX; sp++) {
    snprintf(path, sizeof path, "%s/tp.own.sp%d", dir, sp); load_kv(path, &tp_own_sp[sp]);
    snprintf(path, sizeof path, "%s/dp.own.sp%d", dir, sp); load_kv(path, &dp_own_sp[sp]);
  }
  /* Pool refill histograms (ask + bid) into a single distribution. */
  struct KV ra, rb;
  snprintf(path, sizeof path, "%s/refill.a.own", dir); load_kv(path, &ra);
  snprintf(path, sizeof path, "%s/refill.b.own", dir); load_kv(path, &rb);
  refill.n = 0;
  for (int i = 0; i < ra.n; i++) {
    refill.k[refill.n] = ra.k[i]; refill.v[refill.n] = ra.v[i]; refill.n++;
  }
  for (int i = 0; i < rb.n && refill.n < TMAX; i++) {
    int found = 0;
    for (int j = 0; j < ra.n; j++)
      if (refill.k[j] == rb.k[i]) { refill.v[j] += rb.v[i]; found = 1; break; }
    if (!found) {
      refill.k[refill.n] = rb.k[i]; refill.v[refill.n] = rb.v[i]; refill.n++;
    }
  }
  compute_diffuse_flag(&refill, &refill_diffuse);
  /* Pool tail params by averaging α and summing f_tail. */
  struct TailParam ta, tb;
  snprintf(path, sizeof path, "%s/refill.a.tail", dir); load_tail_param(path, &ta);
  snprintf(path, sizeof path, "%s/refill.b.tail", dir); load_tail_param(path, &tb);
  if (ta.alpha < 0 && tb.alpha < 0) {
    tail.alpha = 0.5 * (ta.alpha + tb.alpha);
    tail.k_cutoff = (ta.k_cutoff + tb.k_cutoff) / 2;
    tail.f_tail = 0.5 * (ta.f_tail + tb.f_tail);
  } else if (ta.alpha < 0) { tail = ta; }
  else if (tb.alpha < 0)  { tail = tb; }
}

static int open_input(const char *path) {
  input_fp = fopen(path, "rb");
  if (!input_fp) { fprintf(stderr, "onestep: cannot open %s\n", path); return 0; }
  is_events_fmt = 1;
  if (idx_path && session_id >= 0) {
    FILE *sf = fopen(idx_path, "rb");
    if (!sf) { fprintf(stderr, "onestep: cannot open %s\n", idx_path); return 0; }
    int64_t off[2];
    if (fseeko(sf, (off_t)session_id * (off_t)sizeof(int64_t), SEEK_SET) != 0 ||
        fread(off, sizeof(int64_t), 2, sf) != 2) {
      fprintf(stderr, "onestep: idx seek/read failed\n"); return 0;
    }
    fclose(sf);
    if (fseeko(input_fp, off[0], SEEK_SET) != 0) {
      fprintf(stderr, "onestep: seek to session %ld failed\n", session_id); return 0;
    }
    bytes_remaining = (off_t)(off[1] - off[0]);
  }
  return 1;
}

/* ── run modes ────────────────────────────────────────────────────────────── */

static int run_chained(double T, int nstep) {
  struct Row r;
  if (!read_wire(&r, input_fp)) return 1;
  reset_phi = 1;
  simulate(&r, T);
  if (!write_wire(&r, stdout)) return 1;
  reset_phi = 0;
  for (int k = 1; k < nstep; k++) {
    simulate(&r, T);
    if (!write_wire(&r, stdout)) return 1;
  }
  return 0;
}

static int run_seed_reps(double T, int stride, int replications) {
  struct Row r;
  int idx = 0;
  while (read_wire(&r, input_fp)) {
    if (idx++ % stride != 0) continue;
    if (!write_wire(&r, stdout)) return 1;
    for (int p = 0; p < replications; p++) {
      struct Row sim = r;
      simulate(&sim, T);
      if (!write_wire(&sim, stdout)) return 1;
    }
  }
  return 0;
}

int main(int argc, char **argv) {
  const char *dir = "tables";
  double T = 1.0;
  long seed = time(NULL);
  int nstep = 0, stride = 1, replications = 1;

  for (int i = 1; i < argc; i++) {
    if      (!strcmp(argv[i], "-m") && i + 1 < argc) dir = argv[++i];
    else if (!strcmp(argv[i], "-T") && i + 1 < argc) T = atof(argv[++i]);
    else if (!strcmp(argv[i], "-R") && i + 1 < argc) seed = atol(argv[++i]);
    else if (!strcmp(argv[i], "-N") && i + 1 < argc) nstep = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-K") && i + 1 < argc) stride = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-S") && i + 1 < argc) idx_path = argv[++i];
    else if (!strcmp(argv[i], "-M") && i + 1 < argc) hk_loaded = load_hawkes(argv[++i]);
    else if (!strcmp(argv[i], "-D") && i + 1 < argc) events_path = argv[++i];
    else if (!strcmp(argv[i], "-s") && i + 1 < argc) session_id = atol(argv[++i]);
    else if (!strcmp(argv[i], "-j") && i + 1 < argc) replications = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-e")) is_events_fmt = 1;
    else if (!strcmp(argv[i], "-Z")) use_state_mu = 1;
    else if (!strcmp(argv[i], "-U")) use_tail = 1;
    else if (!strcmp(argv[i], "-L") && i + 1 < argc) (void)atoi(argv[++i]);
    else {
      fprintf(stderr, "onestep: unknown arg '%s'\n", argv[i]);
      return 1;
    }
  }
  srand48(seed);

  if (events_path) { if (!open_input(events_path)) return 1; }
  else             { input_fp = stdin; }

  load_tables(dir);

  return (nstep > 0) ? run_chained(T, nstep)
                     : run_seed_reps(T, stride, replications);
}
