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

#define DIFFUSE_THRESHOLD 0.95     /* refill.k=2 mass below this ⇒ sample tail */
#define TAIL_ALPHA_MAX   -1.05     /* need α < this for finite tail integral */
#define TAIL_CAP_FACTOR   3        /* truncate power law at k_max = F·k_cutoff */
#define U_FLOOR           1e-12    /* prevent log(0) in Ogata sampling */
#define SP_LIMIT_ABS      256      /* absolute spread cap (2× max observed real) */

struct Row {
  int32_t aR[NL], bR[NL], aS[NL], bS[NL], aN[NL], bN[NL], y;
  int32_t aF[NL], bF[NL];        /* F: 1 = refill-placeholder level */
};

struct Side {                    /* view into r for one side (ask or bid) */
  int32_t *R, *N, *S, *F;
};

struct KV { int n; double k[TMAX]; double v[TMAX]; };

struct TailParam { double alpha; int k_cutoff; double f_tail; };

static FILE *input_fp = NULL;
static int is_events_fmt = 0;   /* set by open_input() when -D given */
static off_t bytes_remaining = -1;
static const char *events_path = NULL, *idx_path = NULL;
static long session_id = -1;

static struct KV tp_a[IMB_BINS], tp_b[IMB_BINS];
static struct KV tm_q_a[IMB_BINS], tm_q_b[IMB_BINS];
static struct KV tm_c_a[IMB_BINS], tm_c_b[IMB_BINS];
static struct KV dp_a[IMB_BINS], dp_b[IMB_BINS];
static struct KV dm_a[IMB_BINS], dm_b[IMB_BINS];
static struct KV n_imb[IMB_BINS];

/* Queue-Reactive tables (Huang-Lehalle-Rosenbaum 2015): rate = f(sp, n_own).
 * Event rates conditional on OWN side's top queue size plus spread.
 * When loaded via `-q <dir>`, these override the imb-conditioned tables. */
enum { N0_MAX = 32, QR_SP_MAX = 64, OPP_MAX = 4 };
static double qr_tp_a[QR_SP_MAX][N0_MAX], qr_tp_b[QR_SP_MAX][N0_MAX];
static double qr_tm_a[QR_SP_MAX][N0_MAX], qr_tm_b[QR_SP_MAX][N0_MAX];
static double qr_dp_a[QR_SP_MAX][N0_MAX], qr_dp_b[QR_SP_MAX][N0_MAX];
static double qr_dm_a[QR_SP_MAX][N0_MAX], qr_dm_b[QR_SP_MAX][N0_MAX];
static int qr_loaded = 0;

static int qr_hk_loaded = 0;  /* Hawkes loaded on top of QR (Wu2019 hybrid) */
static int hk_additive = 0;   /* μ=0 everywhere → additive residual: rate += Σα·φ */

/* QR2: adds opposite-queue bucket to tp/tm rates (HLR's 3-D conditioning).
 * Captures directional asymmetry QR misses (§14 drift at moderate imb). */
static double qr2_tp_a[QR_SP_MAX][N0_MAX][OPP_MAX];
static double qr2_tp_b[QR_SP_MAX][N0_MAX][OPP_MAX];
static double qr2_tm_a[QR_SP_MAX][N0_MAX][OPP_MAX];
static double qr2_tm_b[QR_SP_MAX][N0_MAX][OPP_MAX];
static int qr2_loaded = 0;

static int opp_bucket_of(int n) {
  if (n == 0) return 0;
  if (n <= 2) return 1;
  if (n <= 5) return 2;
  return 3;
}

static struct KV tp_own, dp_own;
static struct KV tp_own_sp[SP_MAX], dp_own_sp[SP_MAX];
static struct KV refill;              /* pooled ask+bid refill histogram */
static int refill_diffuse = 0;

static struct TailParam tail = {0.0, 0, 0.0};   /* pooled tail params */

static double hk_beta = 0.05;
static double hk_mu[N_HAWKES], hk_alpha[N_HAWKES][N_HAWKES];
static double hk_phi[N_HAWKES], hk_phi_stat[N_HAWKES];
static double hk_lambda_stat[N_HAWKES];    /* stationary Hawkes intensity λ_c */

/* Real-history φ: accumulated from pre-seed events, decayed in real time.
 * Pre-warms hk_phi at each seed with data-conditioned clustering state. */
static double real_phi[N_HAWKES];
static int32_t real_last_t = 0;
static int real_phi_init = 0;

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

/* Map events-format raw type (+ pre-event N[0]) to the 6-D pooled index used
 * by the Hawkes fit.  Mirrors preproc.c's tm-split by pre-event queue size. */
static int pooled_event_type(int32_t raw_type, int32_t aN0, int32_t bN0) {
  if (raw_type == 0 || raw_type == 1) return 0;              /* tp */
  if (raw_type == 2) return aN0 > 1 ? 1 : 2;                 /* tm_a → q or c */
  if (raw_type == 3) return bN0 > 1 ? 1 : 2;                 /* tm_b */
  if (raw_type == 4 || raw_type == 5) return 3;              /* dp */
  if (raw_type == 6 || raw_type == 7) return 4;              /* dm */
  if (raw_type >= 9)                  return 5;              /* hp */
  return -1;
}

static void real_phi_decay_to(int32_t t_target) {
  int32_t dt = t_target - real_last_t;
  if (dt <= 0) return;
  double dec = exp(-hk_beta * dt);
  for (int j = 0; j < N_HAWKES; j++) real_phi[j] *= dec;
  real_last_t = t_target;
}

static void real_phi_add_event(int32_t t_ev, int c) {
  if (!real_phi_init) {
    memcpy(real_phi, hk_phi_stat, sizeof real_phi);
    real_last_t = t_ev;
    real_phi_init = 1;
  }
  real_phi_decay_to(t_ev);
  real_phi[c] += 1.0;
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
      /* Accumulate real event → real_phi for warm-start at next seed.
       * Only QR-without-Hawkes skips this (pure memoryless). */
      if (!qr_loaded || qr_hk_loaded) {
        int c = pooled_event_type(w[0], w[5 + 32], w[5 + 40]);
        if (c >= 0) real_phi_add_event(w[1], c);
      }
    }
    if ((!qr_loaded || qr_hk_loaded) && real_phi_init) real_phi_decay_to(w[1]);
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

/* Linear interpolation inside the sampled range.  Past t->k[n-1] the value
 * is kept flat (training-edge estimate) — this is the minimum-risk choice
 * for rates that may extrapolate to 0 (e.g. tm at wide sp).  Callers that
 * want positive extrapolation (tp close-gap) use lookup_tail_max below. */
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

/* Like lookup(), but past the sampled range use the maximum value observed
 * over the top quarter of sp bins.  Prevents edge-of-training zeros from
 * creating dead-zone fixed points in the sim. */
static double lookup_tail_max(struct KV *t, double k) {
  if (t->n == 0) return 0.0;
  if (k < t->k[t->n - 1]) return lookup(t, k);
  double m = 0;
  int start = (t->n * 3) / 4;
  for (int i = start; i < t->n; i++) if (t->v[i] > m) m = t->v[i];
  return m;
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

/* Return the sp-conditional table at (or closest-below) the given spread.
 * Beyond the last populated sp, reuse the largest one rather than falling
 * back to the pooled (sp-agnostic) dist — the latter has almost no mass
 * at large jumps, so the sim can't close a wide gap in one event. */
static struct KV *pick_sp_kv(struct KV *sp_tbl, int32_t sp, struct KV *fallback) {
  if (sp < 0) sp = 0;
  if (sp >= SP_MAX) sp = SP_MAX - 1;
  for (int s = sp; s >= 0; s--)
    if (sp_tbl[s].n > 0) return &sp_tbl[s];
  for (int s = sp + 1; s < SP_MAX; s++)
    if (sp_tbl[s].n > 0) return &sp_tbl[s];
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
  /* Heavy-tail branch auto-engages when refill.*.tail files exist
   * (f_tail > 0 after load_tail_param).  Otherwise fall back to the
   * empirical refill histogram, or TICK if that's also absent. */
  if (tail.f_tail > 0.0 && drand48() < tail.f_tail) return sample_tail(&tail);
  int32_t dist = (refill_diffuse && refill.n > 0) ? (int32_t)sample_dist(&refill) : TICK;
  return dist <= 0 ? TICK : dist;
}

/* ── Hawkes params (6-D single-exponential kernel) ────────────────────────── */

/* Reads single-β Hawkes params. File format:
 *   beta  0           <β>
 *   mu    <c>         <μ_c>
 *   alpha <c> <j>     <α_{c,j}>
 *
 * Stationary λ solves (I − B)·λ = μ where B[c,j] = α[c,j]/β.
 * Then E[φ_j] = λ_j / β. */
static int load_hawkes(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) return 0;
  memset(hk_mu, 0, sizeof hk_mu);
  memset(hk_alpha, 0, sizeof hk_alpha);
  char tag[32]; double v; int c, j, kk;
  while (fscanf(f, "%31s", tag) == 1) {
    if (!strcmp(tag, "beta")) {
      if (fscanf(f, "%d %lf", &kk, &v) != 2) break;
      if (kk == 0) hk_beta = v;
    } else if (!strcmp(tag, "mu")) {
      if (fscanf(f, "%d %lf", &c, &v) != 2) break;
      if (c >= 0 && c < N_HAWKES) hk_mu[c] = v;
    } else if (!strcmp(tag, "alpha")) {
      if (fscanf(f, "%d %d %lf", &c, &j, &v) != 3) break;
      if (c >= 0 && c < N_HAWKES && j >= 0 && j < N_HAWKES)
        hk_alpha[c][j] = v;
    } else break;
  }
  fclose(f);

  /* Detect additive mode: if μ≡0 the params file encodes a residual-fit
   * Hawkes for Wu-Rambaldi additive form (rate = λ_QR + Σα·φ).  No λ_stat
   * solve, no φ_stat warm-up — φ starts at zero and tracks events. */
  double mu_sum = 0;
  for (int c = 0; c < N_HAWKES; c++) mu_sum += hk_mu[c];
  hk_additive = (mu_sum == 0.0);
  if (hk_additive) {
    for (int jj = 0; jj < N_HAWKES; jj++) {
      hk_lambda_stat[jj] = 0.0;
      hk_phi_stat[jj] = 0.0;
      hk_phi[jj] = 0.0;
    }
    return 1;
  }

  /* Multiplicative form (μ>0): stationary λ solves (I − α/β)·λ = μ. */
  for (int c = 0; c < N_HAWKES; c++) {
    double lam = hk_mu[c];
    for (int j = 0; j < N_HAWKES; j++)
      lam += hk_alpha[c][j] * hk_mu[j] / hk_beta;   /* first-order approx */
    hk_lambda_stat[c] = lam;
    hk_phi_stat[c] = lam / hk_beta;
    hk_phi[c] = hk_phi_stat[c];
  }
  /* Fixed-point polish for the (I − α/β)·λ = μ system, 10 iters suffices. */
  for (int it = 0; it < 10; it++) {
    double next[N_HAWKES];
    for (int c = 0; c < N_HAWKES; c++) {
      double s = hk_mu[c];
      for (int j = 0; j < N_HAWKES; j++)
        s += hk_alpha[c][j] * hk_lambda_stat[j] / hk_beta;
      next[c] = s;
    }
    for (int c = 0; c < N_HAWKES; c++) {
      hk_lambda_stat[c] = next[c];
      hk_phi_stat[c] = next[c] / hk_beta;
      hk_phi[c] = hk_phi_stat[c];
    }
  }
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

/* tm_q (queue decrement) and tm_c (cascade) have very different dynamics;
 * using the total tm rate for both overfires cascades at wide sp by ~4×.
 * Asymmetric extrapolation: tp uses tail-max past training range (closes
 * wide-spread gaps reliably); tm_q/tm_c/dp/dm use flat-last (don't
 * manufacture widening events in the extrapolation zone). */
static double pool_rate(int type, int im, int32_t sp) {
  switch (type) {
    case EV_TP:   return lookup_tail_max(&tp_a[im], sp) + lookup_tail_max(&tp_b[im], sp);
    case EV_TM_Q: return lookup(&tm_q_a[im], sp) + lookup(&tm_q_b[im], sp);
    case EV_TM_C: return lookup(&tm_c_a[im], sp) + lookup(&tm_c_b[im], sp);
    case EV_DP:   return lookup(&dp_a[im], sp) + lookup(&dp_b[im], sp);
    case EV_DM:   return lookup(&dm_a[im], sp) + lookup(&dm_b[im], sp);
    default:      return 0;
  }
}

/* tm_q requires a side with N[0]>1; tm_c requires a side with N[0]=1. */
static int tm_q_available(struct Row *r) { return r->aN[0] > 1 || r->bN[0] > 1; }
static int tm_c_available(struct Row *r) { return r->aN[0] == 1 || r->bN[0] == 1; }

static double qr_lookup(double grid[QR_SP_MAX][N0_MAX], int sp, int n0);
static double qr2_lookup(double grid[QR_SP_MAX][N0_MAX][OPP_MAX],
                         int sp, int n0, int opp);

/* QR compute_rates: Queue-Reactive Poisson rates, no Hawkes.  Rates keyed by
 * (sp, own_n0) for dp/dm; (sp, own_n0, opp_bucket) for tp/tm when qr2 loaded.
 * The opp_bucket conditioning fixes HLR's cross-queue asymmetry that pure
 * QR missed at moderate-imb states (§14 drift). */
static double compute_rates_qr(struct Row *r, double rates[N_HAWKES]) {
  int32_t sp = r->aR[0] - r->bR[0];
  int an = r->aN[0], bn = r->bN[0];
  int opp_a = opp_bucket_of(bn);   /* opp of ask = bid's n0 */
  int opp_b = opp_bucket_of(an);   /* opp of bid = ask's n0 */
  double tp, tm;
  if (qr2_loaded) {
    tp = qr2_lookup(qr2_tp_a, sp, an, opp_a) + qr2_lookup(qr2_tp_b, sp, bn, opp_b);
    tm = qr2_lookup(qr2_tm_a, sp, an, opp_a) + qr2_lookup(qr2_tm_b, sp, bn, opp_b);
  } else {
    tp = qr_lookup(qr_tp_a, sp, an) + qr_lookup(qr_tp_b, sp, bn);
    tm = qr_lookup(qr_tm_a, sp, an) + qr_lookup(qr_tm_b, sp, bn);
  }
  double dp = qr_lookup(qr_dp_a, sp, an) + qr_lookup(qr_dp_b, sp, bn);
  double dm = qr_lookup(qr_dm_a, sp, an) + qr_lookup(qr_dm_b, sp, bn);
  /* tm splits by state: tm_c needs n0==1 on some side; tm_q needs n0>1.
   * Attribute tm mass to the available lane; rate is a lane selector. */
  rates[EV_TP]   = tp;
  rates[EV_TM_Q] = (an > 1 || bn > 1) ? tm : 0;
  rates[EV_TM_C] = (an == 1 || bn == 1) ? tm : 0;
  /* If both lanes available, split tm proportionally (heuristic: by n=1 side). */
  if (rates[EV_TM_Q] > 0 && rates[EV_TM_C] > 0) {
    double c_frac = ((an == 1) + (bn == 1)) / 2.0;
    rates[EV_TM_C] = tm * c_frac;
    rates[EV_TM_Q] = tm * (1.0 - c_frac);
  }
  rates[EV_DP]   = dp;
  rates[EV_DM]   = dm;
  rates[EV_HP]   = 0;

  /* Wu-Rambaldi-Muzy-Bacry 2019: QR baseline + Hawkes clustering multiplier.
   * When Hawkes params are loaded alongside QR, rate = λ_QR(s) × λ_H(t)/E[λ_H].
   * At stationarity φ=E[φ] → multiplier=1, rate=λ_QR (state-only).
   * During a burst → multiplier>1 → more events.  Captures temporal
   * self-excitation on top of memoryless state-conditional rates. */
  if (qr_hk_loaded) {
    if (hk_additive) {
      /* Wu-Rambaldi additive residual: rate += Σ_j α_{c,j} φ_j. */
      for (int c = 0; c < N_HAWKES; c++) {
        double add = 0;
        for (int j = 0; j < N_HAWKES; j++) add += hk_alpha[c][j] * hk_phi[j];
        rates[c] += add;
        if (rates[c] < 0) rates[c] = 0;
      }
    } else {
      for (int c = 0; c < N_HAWKES; c++) {
        double lam_curr = hk_mu[c];
        for (int j = 0; j < N_HAWKES; j++) lam_curr += hk_alpha[c][j] * hk_phi[j];
        double m = (hk_lambda_stat[c] > 0) ? lam_curr / hk_lambda_stat[c] : 1.0;
        if (m < 0) m = 0;
        rates[c] *= m;
      }
    }
  }

  double total = 0;
  for (int c = 0; c < N_HAWKES; c++) total += rates[c];
  return total;
}

/* Multiplicative rate: λ_c = pool_rate_c(state) × (λ_c^Hawkes / λ_c^stat).
 *
 *   λ_c^Hawkes = μ_c + Σ_j α[c,j]·φ_j      — current Hawkes intensity
 *   λ_c^stat   = μ_c + Σ_j α[c,j]·(λ_j/β)  — stationary Hawkes intensity
 *
 * Decomposes cleanly: pool_rate sets the state-conditional magnitude; the
 * ratio modulates it by how hot/cold recent history is.  At stationarity
 * (φ = E[φ]), ratio = 1 and rate = pool_rate.  During a burst, ratio > 1.
 * During a lull, ratio < 1.  No double-counting — state and temporal
 * clustering are now independent concerns.
 *
 * State-gates drop tm_q / tm_c when their book configuration is absent. */
static double compute_rates(struct Row *r, double rates[N_HAWKES]) {
  int32_t sp = r->aR[0] - r->bR[0];
  int im = imb_bin(r->aN[0], r->bN[0], r->aN[1], r->bN[1]);
  for (int c = 0; c < N_HAWKES; c++) {
    double lam_curr = hk_mu[c];
    for (int j = 0; j < N_HAWKES; j++)
      lam_curr += hk_alpha[c][j] * hk_phi[j];
    double mult = (hk_lambda_stat[c] > 0) ? lam_curr / hk_lambda_stat[c] : 1.0;
    double base = (c < N_VIS) ? pool_rate(c, im, sp) : hk_mu[c];
    rates[c] = base * mult;
    if (rates[c] < 0) rates[c] = 0;
  }
  if (!tm_q_available(r)) rates[EV_TM_Q] = 0;
  if (!tm_c_available(r)) rates[EV_TM_C] = 0;

  double total = 0;
  for (int c = 0; c < N_HAWKES; c++) total += rates[c];
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
  int32_t sp = r->aR[0] - r->bR[0];
  double ra, rb;
  if (qr_loaded) {
    int opp_a = opp_bucket_of(r->bN[0]);
    int opp_b = opp_bucket_of(r->aN[0]);
    /* tp/tm: use qr2 (3-D) if loaded, else qr (2-D).  dp/dm always qr. */
    switch (type) {
      case EV_TP:
        if (qr2_loaded) {
          ra = qr2_lookup(qr2_tp_a, sp, r->aN[0], opp_a);
          rb = qr2_lookup(qr2_tp_b, sp, r->bN[0], opp_b);
        } else {
          ra = qr_lookup(qr_tp_a, sp, r->aN[0]);
          rb = qr_lookup(qr_tp_b, sp, r->bN[0]);
        } break;
      case EV_TM_Q: case EV_TM_C:
        if (qr2_loaded) {
          ra = qr2_lookup(qr2_tm_a, sp, r->aN[0], opp_a);
          rb = qr2_lookup(qr2_tm_b, sp, r->bN[0], opp_b);
        } else {
          ra = qr_lookup(qr_tm_a, sp, r->aN[0]);
          rb = qr_lookup(qr_tm_b, sp, r->bN[0]);
        } break;
      case EV_DP:   ra = qr_lookup(qr_dp_a, sp, r->aN[0]);
                    rb = qr_lookup(qr_dp_b, sp, r->bN[0]); break;
      case EV_DM:   ra = qr_lookup(qr_dm_a, sp, r->aN[0]);
                    rb = qr_lookup(qr_dm_b, sp, r->bN[0]); break;
      default:      return drand48() < 0.5 ? 0 : 1;
    }
  } else {
    int im = imb_bin(r->aN[0], r->bN[0], r->aN[1], r->bN[1]);
    struct KV *a, *b;
    switch (type) {
      case EV_TP:                  a = &tp_a[im]; b = &tp_b[im]; break;
      case EV_TM_Q:                a = &tm_q_a[im]; b = &tm_q_b[im]; break;
      case EV_TM_C:                a = &tm_c_a[im]; b = &tm_c_b[im]; break;
      case EV_DP:                  a = &dp_a[im]; b = &dp_b[im]; break;
      case EV_DM:                  a = &dm_a[im]; b = &dm_b[im]; break;
      default:     return drand48() < 0.5 ? 0 : 1;
    }
    ra = lookup(a, sp); rb = lookup(b, sp);
  }
  if (ra + rb <= 0) return drand48() < 0.5 ? 0 : 1;
  return drand48() * (ra + rb) < ra ? 0 : 1;
}

/* ── Ogata-thinned simulation loop ────────────────────────────────────────── */

/* Absolute bad-state guard, independent of starting sp.  Previously
 * sp_limit = sp0 · MULT + OFFSET was computed fresh each simulate() call,
 * so in chained mode the cap grew with the book's drift — the check
 * became toothless once sp wandered into the dead zone.  Using a fixed
 * multiple of the max real spread keeps the sim bounded. */
static int bad_state(struct Row *r) {
  int32_t sp = r->aR[0] - r->bR[0];
  if (r->aN[0] == 0 || r->bN[0] == 0) { r->y = 1; return 1; }
  if (sp > SP_LIMIT_ABS || sp <= 0)   { r->y = 2; return 1; }
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

/* Exponential decay of all φ_j by dt with single β. */
static void decay_phi(double dt) {
  double dec = exp(-hk_beta * dt);
  for (int j = 0; j < N_HAWKES; j++) hk_phi[j] *= dec;
}

/* Pick event type by inverse CDF on rates[0..n). */
static int sample_event(const double *rates, int n, double total) {
  double u = drand48() * total, cum = 0;
  for (int k = 0; k < n; k++) {
    cum += rates[k];
    if (u < cum) return k;
  }
  return n - 1;
}

/* Ogata-thinned simulation: advance (r, t) until t reaches T (or bad state).
 * Each iteration: draw Δt ~ Exp(λ*), decay φ, accept at p=λ(t)/λ*, fire event. */
static void simulate(struct Row *r, double T) {
  double t = 0, rates[N_HAWKES];
  r->y = 0;
  if (reset_phi && (!qr_loaded || qr_hk_loaded))
    memcpy(hk_phi, real_phi_init ? real_phi : hk_phi_stat, sizeof hk_phi);

  while (t < T && !bad_state(r)) {
    double lam_star = qr_loaded ? compute_rates_qr(r, rates)
                                : compute_rates(r, rates);
    if (lam_star <= 0) break;
    double dt = -log(drand48()) / lam_star;
    if (t + dt > T) break;
    if (!qr_loaded || qr_hk_loaded) decay_phi(dt);
    t += dt;

    double lam_now = qr_loaded ? compute_rates_qr(r, rates)
                               : compute_rates(r, rates);
    if (drand48() * lam_star >= lam_now) continue;     /* Ogata reject */
    int pick = sample_event(rates, N_HAWKES, lam_now);
    if (!qr_loaded || qr_hk_loaded) hk_phi[pick] += 1.0;
    if (pick == EV_HP) continue;                       /* HP phantom: φ only */
    fire_event(r, pick, sample_side(pick, r), r->aR[0] - r->bR[0]);
  }
}

/* ── table loading orchestrator ───────────────────────────────────────────── */

/* Build a path as "<dir>/<name>" into buf.  Truncation is silent — caller
 * provides a buffer large enough for the paths we emit. */
static void path_join(char *buf, size_t n, const char *dir, const char *name) {
  snprintf(buf, n, "%s/%s", dir, name);
}

/* Try `global_dir/name` first (if given), else fall back to `local_dir/name`.
 * Rate tables stay per-session; jumps/refill come from a shared pool
 * (pool_jumps.py merges them into tables_common). */
static void load_kv_fallback(const char *gdir, const char *ldir,
                             const char *name, struct KV *t) {
  char path[512];
  if (gdir) {
    path_join(path, sizeof path, gdir, name);
    if (load_kv(path, t)) return;
  }
  path_join(path, sizeof path, ldir, name);
  load_kv(path, t);
}

static void load_tail_fallback(const char *gdir, const char *ldir,
                               const char *name, struct TailParam *tp) {
  char path[512];
  if (gdir) {
    path_join(path, sizeof path, gdir, name);
    load_tail_param(path, tp);
    if (tp->alpha != 0.0 || tp->k_cutoff != 0) return;
  }
  path_join(path, sizeof path, ldir, name);
  load_tail_param(path, tp);
}

/* Load QR2 table: rows are "sp n0 opp rate". */
static int load_qr2_table(const char *path,
                          double grid[QR_SP_MAX][N0_MAX][OPP_MAX]) {
  FILE *f = fopen(path, "r");
  if (!f) return 0;
  int sp, n0, opp, nread = 0;
  double v;
  while (fscanf(f, "%d %d %d %lf", &sp, &n0, &opp, &v) == 4) {
    if (sp >= 0 && sp < QR_SP_MAX && n0 >= 0 && n0 < N0_MAX
        && opp >= 0 && opp < OPP_MAX) {
      grid[sp][n0][opp] = v;
      nread++;
    }
  }
  fclose(f);
  return nread;
}

/* Load QR table: rows are "sp n0 rate".  Fill the 2-D grid; unsampled cells
 * stay zero and fall through to nearest-neighbor at lookup time. */
static int load_qr_table(const char *path, double grid[QR_SP_MAX][N0_MAX]) {
  FILE *f = fopen(path, "r");
  if (!f) return 0;
  int sp, n0, nread = 0;
  double v;
  while (fscanf(f, "%d %d %lf", &sp, &n0, &v) == 3) {
    if (sp >= 0 && sp < QR_SP_MAX && n0 >= 0 && n0 < N0_MAX) {
      grid[sp][n0] = v;
      nread++;
    }
  }
  fclose(f);
  return nread;
}

/* QR2 lookup: 3-D (sp, n_own, opp_bucket). Falls back to QR 2-D if cell
 * has zero samples (nearest-neighbor walk). */
static double qr2_lookup(double grid[QR_SP_MAX][N0_MAX][OPP_MAX],
                         int sp, int n0, int opp) {
  if (sp < 0) sp = 0;  if (sp >= QR_SP_MAX) sp = QR_SP_MAX - 1;
  if (n0 < 0) n0 = 0;  if (n0 >= N0_MAX)    n0 = N0_MAX - 1;
  if (opp < 0) opp = 0; if (opp >= OPP_MAX)  opp = OPP_MAX - 1;
  if (grid[sp][n0][opp] > 0) return grid[sp][n0][opp];
  /* Nearest-neighbor: walk opp → n0 → sp. */
  for (int o = opp; o >= 0; o--)
    if (grid[sp][n0][o] > 0) return grid[sp][n0][o];
  for (int n = n0; n >= 0; n--)
    for (int o = 0; o < OPP_MAX; o++)
      if (grid[sp][n][o] > 0) return grid[sp][n][o];
  for (int s = sp; s >= 0; s--)
    for (int n = 0; n < N0_MAX; n++)
      for (int o = 0; o < OPP_MAX; o++)
        if (grid[s][n][o] > 0) return grid[s][n][o];
  return 0;
}

/* QR rate lookup with nearest-sp / nearest-n0 extrapolation.  Extends the
 * sampled grid into unseen states by clamping to the closest observed cell. */
static double qr_lookup(double grid[QR_SP_MAX][N0_MAX], int sp, int n0) {
  if (sp < 0) sp = 0;  if (sp >= QR_SP_MAX) sp = QR_SP_MAX - 1;
  if (n0 < 0) n0 = 0;  if (n0 >= N0_MAX)    n0 = N0_MAX - 1;
  /* Walk sp backward then n0 backward for a non-zero cell. */
  for (int s = sp; s >= 0; s--) {
    for (int n = n0; n >= 0; n--)
      if (grid[s][n] > 0) return grid[s][n];
  }
  return 0;
}

/* Helper: load "<dir>/<ev>.<side>.imb<im>.rates"; fail hard if missing. */
static void load_imb_table(const char *dir, const char *ev, char side,
                           int im, struct KV *t) {
  char path[512];
  snprintf(path, sizeof path, "%s/%s.%c.imb%d.rates", dir, ev, side, im);
  if (!load_kv(path, t)) {
    fprintf(stderr, "onestep: missing required table %s\n", path);
    exit(1);
  }
}

static void load_tables(const char *dir, const char *gdir) {
  char path[512];
  if (qr_loaded) {
    /* QR mode: load 8 qr.*.rates tables (3-column sp n0 rate). */
    const char *evs[] = {"tp", "tm", "dp", "dm"};
    double (*grids_a[])[N0_MAX] = {qr_tp_a, qr_tm_a, qr_dp_a, qr_dm_a};
    double (*grids_b[])[N0_MAX] = {qr_tp_b, qr_tm_b, qr_dp_b, qr_dm_b};
    for (int i = 0; i < 4; i++) {
      char p[512];
      snprintf(p, sizeof p, "%s/qr.%s.a.rates", dir, evs[i]);
      if (load_qr_table(p, grids_a[i]) == 0) {
        fprintf(stderr, "onestep: missing QR table %s\n", p); exit(1);
      }
      snprintf(p, sizeof p, "%s/qr.%s.b.rates", dir, evs[i]);
      if (load_qr_table(p, grids_b[i]) == 0) {
        fprintf(stderr, "onestep: missing QR table %s\n", p); exit(1);
      }
    }
    /* Optional QR2 (opposite-bucket) tables for tp and tm — if all 4 present,
     * use the 3-D lookup for cross-queue asymmetry. */
    char p[512];
    int qr2_ok = 1;
    snprintf(p, sizeof p, "%s/qr2.tp.a.rates", dir); if (!load_qr2_table(p, qr2_tp_a)) qr2_ok = 0;
    snprintf(p, sizeof p, "%s/qr2.tp.b.rates", dir); if (!load_qr2_table(p, qr2_tp_b)) qr2_ok = 0;
    snprintf(p, sizeof p, "%s/qr2.tm.a.rates", dir); if (!load_qr2_table(p, qr2_tm_a)) qr2_ok = 0;
    snprintf(p, sizeof p, "%s/qr2.tm.b.rates", dir); if (!load_qr2_table(p, qr2_tm_b)) qr2_ok = 0;
    qr2_loaded = qr2_ok;
    /* Jump dists + refill still needed for event-effect sampling. */
    load_kv_fallback(gdir, dir, "tp.own", &tp_own);
    load_kv_fallback(gdir, dir, "dp.own", &dp_own);
    for (int sp = 0; sp < SP_MAX; sp++) {
      char name[32];
      snprintf(name, sizeof name, "tp.own.sp%d", sp);
      load_kv_fallback(gdir, dir, name, &tp_own_sp[sp]);
      snprintf(name, sizeof name, "dp.own.sp%d", sp);
      load_kv_fallback(gdir, dir, name, &dp_own_sp[sp]);
    }
    struct KV ra = {0}, rb = {0};
    load_kv_fallback(gdir, dir, "refill.a.own", &ra);
    load_kv_fallback(gdir, dir, "refill.b.own", &rb);
    refill.n = 0;
    for (int i = 0; i < ra.n; i++) {
      refill.k[refill.n] = ra.k[i]; refill.v[refill.n] = ra.v[i]; refill.n++;
    }
    for (int i = 0; i < rb.n && refill.n < TMAX; i++) {
      int found = 0;
      for (int j = 0; j < ra.n; j++)
        if (refill.k[j] == rb.k[i]) { refill.v[j] += rb.v[i]; found = 1; break; }
      if (!found) { refill.k[refill.n] = rb.k[i]; refill.v[refill.n] = rb.v[i]; refill.n++; }
    }
    compute_diffuse_flag(&refill, &refill_diffuse);
    return;
  }
  for (int im = 0; im < IMB_BINS; im++) {
    load_imb_table(dir, "tp",   'a', im, &tp_a[im]);
    load_imb_table(dir, "tp",   'b', im, &tp_b[im]);
    load_imb_table(dir, "tm_q", 'a', im, &tm_q_a[im]);
    load_imb_table(dir, "tm_q", 'b', im, &tm_q_b[im]);
    load_imb_table(dir, "tm_c", 'a', im, &tm_c_a[im]);
    load_imb_table(dir, "tm_c", 'b', im, &tm_c_b[im]);
    load_imb_table(dir, "dp",   'a', im, &dp_a[im]);
    load_imb_table(dir, "dp",   'b', im, &dp_b[im]);
    load_imb_table(dir, "dm",   'a', im, &dm_a[im]);
    load_imb_table(dir, "dm",   'b', im, &dm_b[im]);
    snprintf(path, sizeof path, "%s/n.imb%d.rates", dir, im);
    load_kv(path, &n_imb[im]);
  }
  load_kv_fallback(gdir, dir, "tp.own", &tp_own);
  load_kv_fallback(gdir, dir, "dp.own", &dp_own);
  for (int sp = 0; sp < SP_MAX; sp++) {
    char name[32];
    snprintf(name, sizeof name, "tp.own.sp%d", sp);
    load_kv_fallback(gdir, dir, name, &tp_own_sp[sp]);
    snprintf(name, sizeof name, "dp.own.sp%d", sp);
    load_kv_fallback(gdir, dir, name, &dp_own_sp[sp]);
  }
  /* Pool refill histograms (ask + bid) into a single distribution. */
  struct KV ra, rb;
  load_kv_fallback(gdir, dir, "refill.a.own", &ra);
  load_kv_fallback(gdir, dir, "refill.b.own", &rb);
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
  /* Pool tail params by averaging α and f_tail over present sides. */
  struct TailParam ta = {0}, tb = {0};
  load_tail_fallback(gdir, dir, "refill.a.tail", &ta);
  load_tail_fallback(gdir, dir, "refill.b.tail", &tb);
  if (ta.alpha < 0 && tb.alpha < 0) {
    tail.alpha    = 0.5 * (ta.alpha + tb.alpha);
    tail.k_cutoff = (ta.k_cutoff + tb.k_cutoff) / 2;
    tail.f_tail   = 0.5 * (ta.f_tail + tb.f_tail);
  } else if (ta.alpha < 0) tail = ta;
  else if (tb.alpha < 0)   tail = tb;
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
  const char *gdir = NULL;   /* -g <common_dir>: pooled jump/refill tables */
  double T = 1.0;
  long seed = time(NULL);
  int nstep = 0, stride = 1, replications = 1;

  for (int i = 1; i < argc; i++) {
    if      (!strcmp(argv[i], "-m") && i + 1 < argc) dir = argv[++i];
    else if (!strcmp(argv[i], "-g") && i + 1 < argc) gdir = argv[++i];
    else if (!strcmp(argv[i], "-T") && i + 1 < argc) T = atof(argv[++i]);
    else if (!strcmp(argv[i], "-R") && i + 1 < argc) seed = atol(argv[++i]);
    else if (!strcmp(argv[i], "-N") && i + 1 < argc) nstep = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-K") && i + 1 < argc) stride = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-S") && i + 1 < argc) idx_path = argv[++i];
    else if (!strcmp(argv[i], "-Q")) {
      /* Enable Queue-Reactive mode: per-session qr.*.rates tables, pure Poisson. */
      qr_loaded = 1;
    }
    else if (!strcmp(argv[i], "-M") && i + 1 < argc) {
      if (!load_hawkes(argv[++i])) {
        fprintf(stderr, "onestep: failed to load %s\n", argv[i]);
        return 1;
      }
      qr_hk_loaded = 1;
    }
    else if (!strcmp(argv[i], "-D") && i + 1 < argc) events_path = argv[++i];
    else if (!strcmp(argv[i], "-s") && i + 1 < argc) session_id = atol(argv[++i]);
    else if (!strcmp(argv[i], "-j") && i + 1 < argc) replications = atoi(argv[++i]);
    else {
      fprintf(stderr, "onestep: unknown arg '%s'\n", argv[i]);
      return 1;
    }
  }
  srand48(seed);

  if (events_path) { if (!open_input(events_path)) return 1; }
  else             { input_fp = stdin; }

  load_tables(dir, gdir);

  return (nstep > 0) ? run_chained(T, nstep)
                     : run_seed_reps(T, stride, replications);
}
