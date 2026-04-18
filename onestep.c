#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { nl = 8, TMAX = 500, SP_MAX = 64, HMAX = 2048, HIDE = 64, IMB_BINS = 6 };

static int imb_bin(int32_t aN0, int32_t bN0, int32_t aN1, int32_t bN1) {
  int b0;
  int64_t s = (int64_t)aN0 + bN0;
  int64_t d = (int64_t)aN0 - bN0;
  if (s == 0)
    b0 = 1;
  else if (d * 5 < -s)
    b0 = 0;
  else if (d * 5 > s)
    b0 = 2;
  else
    b0 = 1;
  int s1 = (aN1 > bN1) ? 1 : 0;
  return b0 * 2 + s1;
}
struct Row {
  int32_t aR[nl], bR[nl], aS[nl], bS[nl], aN[nl], bN[nl], y;
  int32_t aF[nl], bF[nl];
};

static struct Row hbuf[HMAX];
static int hn = 0, hh = 0, lookback = 0, nr_window = 0;
static int32_t ahide[HIDE], bhide[HIDE];
static int ahn = 0, bhn = 0;

static void hist_push(struct Row *r) {
  hbuf[hh] = *r;
  hh = (hh + 1) % HMAX;
  if (hn < HMAX)
    hn++;
}

static double recent_nr(int target_events) {
  int max_scan = hn < 2 ? 0 : hn - 1;
  if (max_scan < 1)
    return -1.0;
  int i, events = 0, total = 0;
  for (i = 1; i <= max_scan; i++) {
    int i0 = (hh - i - 1 + HMAX) % HMAX;
    int i1 = (hh - i + HMAX) % HMAX;
    if (memcmp(hbuf[i0].aR, hbuf[i1].aR, 48 * sizeof(int32_t)) != 0)
      events++;
    total++;
    if (events >= target_events)
      break;
  }
  if (total < 10)
    return -1.0;
  return (double)(total - events) / total;
}

static int cmp_asc(const void *a, const void *b) {
  int32_t x = *(int32_t *)a, y = *(int32_t *)b;
  return x < y ? -1 : x > y ? 1 : 0;
}
static int cmp_desc(const void *a, const void *b) {
  return cmp_asc(b, a);
}

static void reveal(struct Row *cur) {
  int side, k, i, j;
  int32_t cand[HMAX * nl];
  int n_cand;
  int kmax = lookback < hn - 1 ? lookback : hn - 1;
  ahn = bhn = 0;
  if (kmax < 1)
    return;
  for (side = 0; side < 2; side++) {
    n_cand = 0;
    for (k = 1; k <= kmax; k++) {
      struct Row *r0 = &hbuf[(hh - k - 2 + HMAX) % HMAX];
      struct Row *r1 = &hbuf[(hh - k - 1 + HMAX) % HMAX];
      int32_t R0[nl], R1[nl];
      int n0 = 0, n1 = 0;
      int32_t *sR0 = side ? r0->bR : r0->aR;
      int32_t *sN0 = side ? r0->bN : r0->aN;
      int32_t *sR1 = side ? r1->bR : r1->aR;
      int32_t *sN1 = side ? r1->bN : r1->aN;
      for (i = 0; i < nl && sN0[i]; i++)
        R0[n0++] = side ? -sR0[i] : sR0[i];
      for (i = 0; i < nl && sN1[i]; i++)
        R1[n1++] = side ? -sR1[i] : sR1[i];
      if (n0 == 0)
        continue;
      i = 0;
      j = 0;
      while (i < n0 && j < n1) {
        if (R1[j] < R0[i])
          j++;
        else if (R1[j] == R0[i]) {
          i++;
          j++;
        } else
          cand[n_cand++] = side ? -R0[i++] : R0[i++];
      }
      while (i < n0)
        cand[n_cand++] = side ? -R0[i++] : R0[i++];
    }
    if (n_cand == 0)
      continue;
    int32_t *R = side ? cur->bR : cur->aR;
    int cur_n = 0;
    for (i = 0; i < nl && (side ? cur->bN[i] : cur->aN[i]); i++)
      cur_n++;
    int32_t bottom = R[cur_n - 1];
    qsort(cand, n_cand, sizeof *cand, side ? cmp_desc : cmp_asc);
    int32_t *hide = side ? bhide : ahide;
    int *hn_ = side ? &bhn : &ahn;
    int32_t last = bottom;
    for (i = 0; i < n_cand && *hn_ < HIDE; i++) {
      int32_t p = cand[i];
      if (side ? p >= last : p <= last)
        continue;
      hide[(*hn_)++] = p;
      last = p;
    }
  }
}

static int hide_pop(int side, int32_t bottom, int32_t *out) {
  int32_t *h = side ? bhide : ahide;
  int *n = side ? &bhn : &ahn;
  int i, j;
  for (i = 0; i < *n; i++) {
    if (side ? h[i] < bottom : h[i] > bottom) {
      *out = h[i];
      for (j = i + 1; j < *n; j++)
        h[j - 1] = h[j];
      (*n)--;
      return 1;
    }
  }
  return 0;
}

static int read_wire(struct Row *r, FILE *f) {
  int32_t w[49];
  if (fread(w, sizeof w, 1, f) != 1)
    return 0;
  memset(r, 0, sizeof *r);
  memcpy(r->aR, &w[0],  nl * 4);
  memcpy(r->bR, &w[8],  nl * 4);
  memcpy(r->aS, &w[16], nl * 4);
  memcpy(r->bS, &w[24], nl * 4);
  memcpy(r->aN, &w[32], nl * 4);
  memcpy(r->bN, &w[40], nl * 4);
  r->y = w[48];
  return 1;
}
static int write_wire(struct Row *r, FILE *f) {
  int32_t w[49];
  memcpy(&w[0],  r->aR, nl * 4);
  memcpy(&w[8],  r->bR, nl * 4);
  memcpy(&w[16], r->aS, nl * 4);
  memcpy(&w[24], r->bS, nl * 4);
  memcpy(&w[32], r->aN, nl * 4);
  memcpy(&w[40], r->bN, nl * 4);
  w[48] = r->y;
  return fwrite(w, sizeof w, 1, f) == 1;
}

struct KV {
  int n;
  double k[TMAX];
  double v[TMAX];
};

static int load_kv(char *path, struct KV *t) {
  t->n = 0;
  FILE *f = fopen(path, "r");
  if (f == NULL)
    return 0;
  while (t->n < TMAX &&
         fscanf(f, "%lf %lf", &t->k[t->n], &t->v[t->n]) == 2)
    t->n++;
  fclose(f);
  return t->n;
}

static double lookup(struct KV *t, double k) {
  int i;
  if (t->n == 0)
    return 0.0;
  if (k <= t->k[0])
    return t->v[0];
  if (k >= t->k[t->n - 1])
    return t->v[t->n - 1];
  for (i = 1; i < t->n; i++)
    if (k <= t->k[i]) {
      double a = (k - t->k[i - 1]) / (t->k[i] - t->k[i - 1]);
      return t->v[i - 1] * (1 - a) + t->v[i] * a;
    }
  return t->v[t->n - 1];
}

static double sample_dist(struct KV *t) {
  int i;
  double total = 0, u, cum = 0;
  for (i = 0; i < t->n; i++)
    total += t->v[i];
  if (total <= 0)
    return 0;
  u = drand48() * total;
  for (i = 0; i < t->n; i++) {
    cum += t->v[i];
    if (u <= cum)
      return t->k[i];
  }
  return t->k[t->n - 1];
}

static struct KV tp_rates, tm_rates, dp_rates, dm_rates, r_rates, n_rates;
static struct KV tp_a[IMB_BINS], tp_b[IMB_BINS];
static struct KV tm_a[IMB_BINS], tm_b[IMB_BINS];
static struct KV dp_a[IMB_BINS], dp_b[IMB_BINS];
static struct KV dm_a[IMB_BINS], dm_b[IMB_BINS];
static struct KV n_imb[IMB_BINS];
static int imb_tables_loaded = 0;
static int queue_mode = 0;
static struct KV q_mu_a[IMB_BINS], q_mu_b[IMB_BINS];
static struct KV q_nu_a[IMB_BINS], q_nu_b[IMB_BINS];
static int qr_mode = 0;
enum { N0_BINS = 32, SP_BINS = 64 };
static double qr_rate[5][2][SP_BINS][N0_BINS];
static double qr_n[SP_BINS][N0_BINS];
static int qr_loaded = 0;
static int load_qr(const char *dir) {
  int k, s, ev;
  char path[512];
  for (ev = 0; ev < 5; ev++)
    for (s = 0; s < 2; s++)
      memset(qr_rate[ev][s], 0, sizeof qr_rate[ev][s]);
  memset(qr_n, 0, sizeof qr_n);
  const char *evs[5] = {"tp", "tm", "dp", "dm", "r"};
  for (ev = 0; ev < 5; ev++) {
    for (s = 0; s < 2; s++) {
      snprintf(path, sizeof path, "%s/qr.%s.%c.rates", dir, evs[ev], s ? 'b' : 'a');
      FILE *f = fopen(path, "r");
      if (!f) return 0;
      int sp, n0;
      double rate;
      while (fscanf(f, "%d %d %lf", &sp, &n0, &rate) == 3) {
        if (sp >= 0 && sp < SP_BINS && n0 >= 0 && n0 < N0_BINS)
          qr_rate[ev][s][sp][n0] = rate;
      }
      fclose(f);
    }
  }
  snprintf(path, sizeof path, "%s/qr.n.rates", dir);
  FILE *f = fopen(path, "r");
  if (!f) return 0;
  int sp, n0;
  double rate;
  while (fscanf(f, "%d %d %lf", &sp, &n0, &rate) == 3) {
    if (sp >= 0 && sp < SP_BINS && n0 >= 0 && n0 < N0_BINS)
      qr_n[sp][n0] = rate;
  }
  fclose(f);
  return 1;
}
static double qr_lookup(int ev, int side, int sp, int n0) {
  if (sp < 0) sp = 0; if (sp >= SP_BINS) sp = SP_BINS - 1;
  if (n0 < 0) n0 = 0; if (n0 >= N0_BINS) n0 = N0_BINS - 1;
  double r = qr_rate[ev][side][sp][n0];
  if (r > 0) return r;
  int d;
  for (d = 1; d < N0_BINS; d++) {
    int lo = n0 - d, hi = n0 + d;
    if (lo >= 0 && qr_rate[ev][side][sp][lo] > 0) return qr_rate[ev][side][sp][lo];
    if (hi < N0_BINS && qr_rate[ev][side][sp][hi] > 0) return qr_rate[ev][side][sp][hi];
  }
  return 0;
}
static int hawkes_on = 0;
static double h_beta = 0.02, h_alpha = 0.2;
static double h_mem_a[4], h_mem_b[4];
static struct KV tp_own, dp_own;
static struct KV tp_own_sp[SP_MAX], dp_own_sp[SP_MAX];

static struct KV *pick_own(struct KV *sp_tbl, int32_t sp, struct KV *fallback) {
  if (sp >= 0 && sp < SP_MAX && sp_tbl[sp].n > 0)
    return &sp_tbl[sp];
  return fallback;
}

static void apply_tp(struct Row *r, int side, int32_t dist) {
  int32_t *R = side ? r->bR : r->aR;
  int32_t *N = side ? r->bN : r->aN;
  int32_t *S = side ? r->bS : r->aS;
  int32_t *F = side ? r->bF : r->aF;
  int k;
  if (dist <= 0) {
    N[0]++;
    S[0]++;
    return;
  }
  int32_t opp = side ? r->aR[0] : r->bR[0];
  int32_t newR = side ? R[0] + dist : R[0] - dist;
  if ((side == 0 && newR <= opp) || (side == 1 && newR >= opp)) {
    newR = side ? opp - 2 : opp + 2;
    if ((side == 0 && newR <= opp) || (side == 1 && newR >= opp)) {
      N[0]++;
      S[0]++;
      return;
    }
  }
  for (k = nl - 1; k > 0; k--) {
    R[k] = R[k - 1];
    N[k] = N[k - 1];
    S[k] = S[k - 1];
    F[k] = F[k - 1];
  }
  R[0] = newR;
  N[0] = 1;
  S[0] = 1;
  F[0] = 0;
}

static void apply_dp(struct Row *r, int side, int32_t dist) {
  int32_t *R = side ? r->bR : r->aR;
  int32_t *N = side ? r->bN : r->aN;
  int32_t *S = side ? r->bS : r->aS;
  int32_t *F = side ? r->bF : r->aF;
  int k;
  if (dist <= 0)
    return;
  int32_t newR = side ? R[0] - dist : R[0] + dist;
  for (k = 1; k < nl; k++) {
    if (N[k] == 0)
      break;
    if (R[k] == newR) {
      if (F[k] == 0) {
        N[k]++;
        S[k]++;
      } else {
        N[k] = 1;
        S[k] = 1;
        F[k] = 0;
      }
      return;
    }
    int past = side ? (newR > R[k]) : (newR < R[k]);
    if (past)
      break;
  }
  if (k == nl)
    return;
  int j;
  for (j = nl - 1; j > k; j--) {
    R[j] = R[j - 1];
    N[j] = N[j - 1];
    S[j] = S[j - 1];
    F[j] = F[j - 1];
  }
  R[k] = newR;
  N[k] = 1;
  S[k] = 1;
  F[k] = 0;
}

static void apply_tm(struct Row *r, int side) {
  int32_t *R = side ? r->bR : r->aR;
  int32_t *N = side ? r->bN : r->aN;
  int32_t *S = side ? r->bS : r->aS;
  int32_t *F = side ? r->bF : r->aF;
  int k;
  if (N[0] == 0)
    return;
  N[0]--;
  if (S[0] > 0)
    S[0]--;
  if (N[0] == 0) {
    for (k = 0; k < nl - 1; k++) {
      R[k] = R[k + 1];
      N[k] = N[k + 1];
      S[k] = S[k + 1];
      F[k] = F[k + 1];
    }
    int32_t hp;
    if (N[nl - 2] > 0 && hide_pop(side, R[nl - 2], &hp)) {
      R[nl - 1] = hp;
      N[nl - 1] = 1;
      S[nl - 1] = 1;
      F[nl - 1] = 0;
    } else if (N[nl - 2] > 0) {
      R[nl - 1] = side ? R[nl - 2] - 2 : R[nl - 2] + 2;
      N[nl - 1] = 1;
      S[nl - 1] = 1;
      F[nl - 1] = 1;
    } else {
      R[nl - 1] = 0;
      N[nl - 1] = 0;
      S[nl - 1] = 0;
      F[nl - 1] = 0;
    }
  }
}

static int apply_dm(struct Row *r, int side) {
  int32_t *R = side ? r->bR : r->aR;
  int32_t *N = side ? r->bN : r->aN;
  int32_t *S = side ? r->bS : r->aS;
  int32_t *F = side ? r->bF : r->aF;
  int total = 0, k;
  for (k = 1; k < nl; k++)
    if (N[k] > 0 && F[k] == 0)
      total += N[k];
  if (total == 0)
    return 0;
  int u = (int)(drand48() * total);
  int sum = 0, pick = 1;
  for (k = 1; k < nl; k++) {
    if (N[k] == 0 || F[k] != 0)
      continue;
    sum += N[k];
    if (u < sum) {
      pick = k;
      break;
    }
  }
  N[pick]--;
  if (S[pick] > 0)
    S[pick]--;
  if (N[pick] == 0) {
    int j;
    for (j = pick; j < nl - 1; j++) {
      R[j] = R[j + 1];
      N[j] = N[j + 1];
      S[j] = S[j + 1];
      F[j] = F[j + 1];
    }
    int32_t hp;
    if (N[nl - 2] > 0 && hide_pop(side, R[nl - 2], &hp)) {
      R[nl - 1] = hp;
      N[nl - 1] = 1;
      S[nl - 1] = 1;
      F[nl - 1] = 0;
    } else if (N[nl - 2] > 0) {
      R[nl - 1] = side ? R[nl - 2] - 2 : R[nl - 2] + 2;
      N[nl - 1] = 1;
      S[nl - 1] = 1;
      F[nl - 1] = 1;
    } else {
      R[nl - 1] = 0;
      N[nl - 1] = 0;
      S[nl - 1] = 0;
      F[nl - 1] = 0;
    }
  }
  return 1;
}

static void simulate(struct Row *r, double T) {
  double t = 0;
  int32_t sp0 = r->aR[0] - r->bR[0];
  int32_t sp_limit = (sp0 > 0 ? sp0 : 2) * 10 + 100;
  r->y = 0;
  if (hawkes_on) {
    memset(h_mem_a, 0, sizeof h_mem_a);
    memset(h_mem_b, 0, sizeof h_mem_b);
  }
  while (t < T) {
    int32_t sp = r->aR[0] - r->bR[0];
    if (r->aN[0] == 0 || r->bN[0] == 0) {
      r->y = 1;
      break;
    }
    if (sp > sp_limit || sp <= 0) {
      r->y = 2;
      break;
    }
    double rates[8];
    double total = 0;
    if (qr_mode && qr_loaded) {
      int n0_a = r->aN[0] < N0_BINS ? r->aN[0] : N0_BINS - 1;
      int n0_b = r->bN[0] < N0_BINS ? r->bN[0] : N0_BINS - 1;
      int sp_c = sp < SP_BINS ? (sp < 0 ? 0 : sp) : SP_BINS - 1;
      rates[0] = qr_lookup(0, 0, sp_c, n0_a);
      rates[1] = qr_lookup(0, 1, sp_c, n0_b);
      rates[2] = qr_lookup(1, 0, sp_c, n0_a);
      rates[3] = qr_lookup(1, 1, sp_c, n0_b);
      rates[4] = qr_lookup(2, 0, sp_c, n0_a);
      rates[5] = qr_lookup(2, 1, sp_c, n0_b);
      rates[6] = qr_lookup(3, 0, sp_c, n0_a);
      rates[7] = qr_lookup(3, 1, sp_c, n0_b);
      int k;
      for (k = 0; k < 8; k++) total += rates[k];
      if (total <= 0) break;
      double nr = qr_n[sp_c][n0_a];
      if (nr > 0 && nr < 1) {
        double target = 1.0 - nr;
        double scale = target / total;
        for (k = 0; k < 8; k++) rates[k] *= scale;
        total = target;
      }
    } else if (queue_mode && imb_tables_loaded) {
      int im = imb_bin(r->aN[0], r->bN[0], r->aN[1], r->bN[1]);
      int32_t aN0c = r->aN[0], bN0c = r->bN[0];
      int32_t aNdc = 0, bNdc = 0;
      int lv;
      for (lv = 1; lv < nl; lv++) { aNdc += r->aN[lv]; bNdc += r->bN[lv]; }
      rates[0] = lookup(&tp_a[im], sp);
      rates[1] = lookup(&tp_b[im], sp);
      rates[2] = lookup(&q_mu_a[im], sp) * aN0c;
      rates[3] = lookup(&q_mu_b[im], sp) * bN0c;
      rates[4] = lookup(&dp_a[im], sp);
      rates[5] = lookup(&dp_b[im], sp);
      rates[6] = lookup(&q_nu_a[im], sp) * aNdc;
      rates[7] = lookup(&q_nu_b[im], sp) * bNdc;
      int k;
      for (k = 0; k < 8; k++) total += rates[k];
      if (total <= 0) break;
      double nr = lookup(&n_imb[im], sp);
      if (nr_window > 0) {
        double nr_loc = recent_nr(nr_window);
        if (nr_loc > 0) nr = nr_loc;
      }
      if (nr > 0 && nr < 1) {
        double target = 1.0 - nr;
        double scale = target / total;
        for (k = 0; k < 8; k++) rates[k] *= scale;
        total = target;
      }
    } else if (imb_tables_loaded) {
      int im = imb_bin(r->aN[0], r->bN[0], r->aN[1], r->bN[1]);
      rates[0] = lookup(&tp_a[im], sp); rates[1] = lookup(&tp_b[im], sp);
      rates[2] = lookup(&tm_a[im], sp); rates[3] = lookup(&tm_b[im], sp);
      rates[4] = lookup(&dp_a[im], sp); rates[5] = lookup(&dp_b[im], sp);
      rates[6] = lookup(&dm_a[im], sp); rates[7] = lookup(&dm_b[im], sp);
      int k;
      for (k = 0; k < 8; k++) total += rates[k];
      if (total <= 0) break;
      double nr = lookup(&n_imb[im], sp);
      if (nr_window > 0) {
        double nr_loc = recent_nr(nr_window);
        if (nr_loc > 0) nr = nr_loc;
      }
      if (nr > 0 && nr < 1) {
        double target = 1.0 - nr;
        double scale = target / total;
        for (k = 0; k < 8; k++) rates[k] *= scale;
        total = target;
      }
    } else {
      double rtp = lookup(&tp_rates, sp);
      double rtm = lookup(&tm_rates, sp);
      double rdp = lookup(&dp_rates, sp);
      double rdm = lookup(&dm_rates, sp);
      total = rtp + rtm + rdp + rdm;
      if (total <= 0) break;
      if (n_rates.n > 0) {
        double nr = lookup(&n_rates, sp);
        if (nr > 0 && nr < 1) {
          double target = 1.0 - nr;
          double scale = target / total;
          rtp *= scale; rtm *= scale; rdp *= scale; rdm *= scale;
          total = target;
        }
      }
      rates[0] = rtp/2; rates[1] = rtp/2;
      rates[2] = rtm/2; rates[3] = rtm/2;
      rates[4] = rdp/2; rates[5] = rdp/2;
      rates[6] = rdm/2; rates[7] = rdm/2;
    }
    if (hawkes_on) {
      int k;
      double ab = h_alpha * h_beta;
      for (k = 0; k < 4; k++) {
        rates[2*k]   += ab * h_mem_a[k];
        rates[2*k+1] += ab * h_mem_b[k];
      }
      total = 0;
      for (k = 0; k < 8; k++) total += rates[k];
      if (total <= 0) break;
    }
    double dt = -log(drand48()) / total;
    if (t + dt > T)
      break;
    if (hawkes_on) {
      double f = exp(-h_beta * dt);
      int k;
      for (k = 0; k < 4; k++) { h_mem_a[k] *= f; h_mem_b[k] *= f; }
    }
    t += dt;
    double u = drand48() * total;
    double cum = 0;
    int pick = 7, k;
    for (k = 0; k < 8; k++) {
      cum += rates[k];
      if (u < cum) { pick = k; break; }
    }
    int side = pick & 1;
    int type = pick >> 1;
    if (hawkes_on) {
      if (side == 0) h_mem_a[type] += 1.0; else h_mem_b[type] += 1.0;
    }
    if (type == 0) {
      apply_tp(r, side,
               (int32_t)sample_dist(pick_own(tp_own_sp, sp, &tp_own)));
    } else if (type == 1) {
      apply_tm(r, side);
    } else if (type == 2) {
      apply_dp(r, side,
               (int32_t)sample_dist(pick_own(dp_own_sp, sp, &dp_own)));
    } else {
      apply_dm(r, side);
    }
  }
}

int main(int argc, char **argv) {
  char *dir = "tables";
  double T = 1.0;
  long seed = time(NULL);
  int nstep = 0, stride = 1;
  int i;
  for (i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-m") && i + 1 < argc)
      dir = argv[++i];
    else if (!strcmp(argv[i], "-T") && i + 1 < argc)
      T = atof(argv[++i]);
    else if (!strcmp(argv[i], "-R") && i + 1 < argc)
      seed = atol(argv[++i]);
    else if (!strcmp(argv[i], "-N") && i + 1 < argc)
      nstep = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-L") && i + 1 < argc)
      lookback = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-S") && i + 1 < argc)
      stride = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-W") && i + 1 < argc)
      nr_window = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-Q"))
      queue_mode = 1;
    else if (!strcmp(argv[i], "-X"))
      qr_mode = 1;
    else if (!strcmp(argv[i], "-H"))
      hawkes_on = 1;
    else if (!strcmp(argv[i], "-P"))
      hawkes_on = 0;
    else if (!strcmp(argv[i], "-A") && i + 1 < argc)
      h_alpha = atof(argv[++i]);
    else if (!strcmp(argv[i], "-B") && i + 1 < argc)
      h_beta = atof(argv[++i]);
    else {
      fprintf(stderr, "onestep.c: error: unknown arg '%s'\n", argv[i]);
      return 1;
    }
  }
  srand48(seed);
  char path[512];
  snprintf(path, sizeof path, "%s/tp.rates", dir);
  load_kv(path, &tp_rates);
  snprintf(path, sizeof path, "%s/tm.rates", dir);
  load_kv(path, &tm_rates);
  snprintf(path, sizeof path, "%s/dp.rates", dir);
  load_kv(path, &dp_rates);
  snprintf(path, sizeof path, "%s/dm.rates", dir);
  load_kv(path, &dm_rates);
  snprintf(path, sizeof path, "%s/r.rates", dir);
  load_kv(path, &r_rates);
  snprintf(path, sizeof path, "%s/n.rates", dir);
  load_kv(path, &n_rates);
  int im;
  if (qr_mode) qr_loaded = load_qr(dir);
  if (queue_mode) {
    for (im = 0; im < IMB_BINS; im++) {
      snprintf(path, sizeof path, "%s/q_mu.a.imb%d.rates", dir, im);
      load_kv(path, &q_mu_a[im]);
      snprintf(path, sizeof path, "%s/q_mu.b.imb%d.rates", dir, im);
      load_kv(path, &q_mu_b[im]);
      snprintf(path, sizeof path, "%s/q_nu.a.imb%d.rates", dir, im);
      load_kv(path, &q_nu_a[im]);
      snprintf(path, sizeof path, "%s/q_nu.b.imb%d.rates", dir, im);
      load_kv(path, &q_nu_b[im]);
    }
  }
  snprintf(path, sizeof path, "%s/tp.own", dir);
  load_kv(path, &tp_own);
  snprintf(path, sizeof path, "%s/dp.own", dir);
  load_kv(path, &dp_own);
  imb_tables_loaded = 1;
  for (im = 0; im < IMB_BINS; im++) {
    snprintf(path, sizeof path, "%s/tp.a.imb%d.rates", dir, im);
    if (!load_kv(path, &tp_a[im])) imb_tables_loaded = 0;
    snprintf(path, sizeof path, "%s/tp.b.imb%d.rates", dir, im);
    if (!load_kv(path, &tp_b[im])) imb_tables_loaded = 0;
    snprintf(path, sizeof path, "%s/tm.a.imb%d.rates", dir, im);
    if (!load_kv(path, &tm_a[im])) imb_tables_loaded = 0;
    snprintf(path, sizeof path, "%s/tm.b.imb%d.rates", dir, im);
    if (!load_kv(path, &tm_b[im])) imb_tables_loaded = 0;
    snprintf(path, sizeof path, "%s/dp.a.imb%d.rates", dir, im);
    if (!load_kv(path, &dp_a[im])) imb_tables_loaded = 0;
    snprintf(path, sizeof path, "%s/dp.b.imb%d.rates", dir, im);
    if (!load_kv(path, &dp_b[im])) imb_tables_loaded = 0;
    snprintf(path, sizeof path, "%s/dm.a.imb%d.rates", dir, im);
    if (!load_kv(path, &dm_a[im])) imb_tables_loaded = 0;
    snprintf(path, sizeof path, "%s/dm.b.imb%d.rates", dir, im);
    if (!load_kv(path, &dm_b[im])) imb_tables_loaded = 0;
    snprintf(path, sizeof path, "%s/n.imb%d.rates", dir, im);
    load_kv(path, &n_imb[im]);
  }
  int sp;
  for (sp = 0; sp < SP_MAX; sp++) {
    snprintf(path, sizeof path, "%s/tp.own.sp%d", dir, sp);
    load_kv(path, &tp_own_sp[sp]);
    snprintf(path, sizeof path, "%s/dp.own.sp%d", dir, sp);
    load_kv(path, &dp_own_sp[sp]);
  }
  struct Row r;
  if (nstep > 0) {
    if (!read_wire(&r, stdin))
      return 1;
    int k;
    for (k = 0; k < nstep; k++) {
      simulate(&r, T);
      if (!write_wire(&r, stdout))
        return 1;
    }
  } else {
    int idx = 0;
    while (read_wire(&r, stdin)) {
      hist_push(&r);
      if (idx++ % stride == 0) {
        reveal(&r);
        if (!write_wire(&r, stdout))
          return 1;
        struct Row sim = r;
        simulate(&sim, T);
        if (!write_wire(&sim, stdout))
          return 1;
      }
    }
  }
  return 0;
}
