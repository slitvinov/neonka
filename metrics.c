/* metrics.c — core sections of compare.py in C.
 *
 * Reads two .raw files (ROW_COLS int32 per row) and prints:
 *   0. flags        1. basics      2. spread distribution
 *   3. nc0 distribution           5. return stats (mean/std/skew/kurt)
 *   6. return ACF   7. mid-price R²                13. queue imbalance
 *  14. E[Δmid | imb_bin]
 *
 * Usage: metrics <specA> <specB>
 *   spec = path[:tag[:stride]]   tag = int (session id) | odd | even
 * Session bounds loaded from sessions.raw next to the .raw file.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

enum { NL = 8, ROW_COLS = 49, W = 14 };

#define AR(s, i)  (s)->rows[(i) * ROW_COLS + 0]
#define BR(s, i)  (s)->rows[(i) * ROW_COLS + 8]
#define AN(s, i)  (s)->rows[(i) * ROW_COLS + 32]
#define AN1(s, i) (s)->rows[(i) * ROW_COLS + 33]
#define BN(s, i)  (s)->rows[(i) * ROW_COLS + 40]
#define Y(s, i)   (s)->rows[(i) * ROW_COLS + 48]

struct Source {
  int32_t *rows;
  long n;
  char label[256];
};

/* ── source loading ────────────────────────────────────────────────────────── */

static int load_source(const char *spec, struct Source *src) {
  char path[512]; long session_id = -1; int row_sel = 0; int stride = 1;
  const char *c1 = strchr(spec, ':');
  if (!c1) {
    strncpy(path, spec, sizeof path - 1); path[sizeof path - 1] = 0;
  } else {
    size_t pl = c1 - spec;
    if (pl >= sizeof path) return 0;
    memcpy(path, spec, pl); path[pl] = 0;
    const char *tag = c1 + 1;
    const char *c2 = strchr(tag, ':');
    char tb[64];
    size_t tl = c2 ? (size_t)(c2 - tag) : strlen(tag);
    if (tl >= sizeof tb) return 0;
    memcpy(tb, tag, tl); tb[tl] = 0;
    if      (!strcmp(tb, "odd"))  row_sel = 1;
    else if (!strcmp(tb, "even")) row_sel = 2;
    else                          session_id = atol(tb);
    if (c2) stride = atoi(c2 + 1);
    if (stride < 1) stride = 1;
  }
  int fd = open(path, O_RDONLY);
  if (fd < 0) { fprintf(stderr, "metrics: cannot open %s\n", path); return 0; }
  struct stat st; if (fstat(fd, &st) != 0) { close(fd); return 0; }
  int32_t *base = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (base == MAP_FAILED) { perror("mmap"); close(fd); return 0; }
  long nrows = st.st_size / (ROW_COLS * (off_t)sizeof(int32_t));
  long lo = 0, hi = nrows;
  if (session_id >= 0) {
    char sess[512];
    const char *slash = strrchr(path, '/');
    int dlen = slash ? (int)(slash - path) + 1 : 0;
    memcpy(sess, path, dlen); strcpy(sess + dlen, "sessions.raw");
    FILE *sf = fopen(sess, "rb");
    if (!sf) { strcpy(sess + dlen, "session.raw"); sf = fopen(sess, "rb"); }
    if (!sf) {
      fprintf(stderr, "metrics: no sessions.raw near %s\n", path);
      munmap(base, st.st_size); close(fd); return 0;
    }
    fseeko(sf, session_id * (off_t)sizeof(int64_t), SEEK_SET);
    int64_t b[2];
    if (fread(b, sizeof(int64_t), 2, sf) != 2) {
      fclose(sf); munmap(base, st.st_size); close(fd); return 0;
    }
    fclose(sf);
    lo = b[0]; hi = b[1];
  }
  long start, step;
  if      (row_sel == 1) { start = lo + 1; step = 2 * stride; }
  else if (row_sel == 2) { start = lo;     step = 2 * stride; }
  else                   { start = lo;     step = stride; }

  long n = (start < hi) ? (hi - start + step - 1) / step : 0;
  if (n <= 0) {
    fprintf(stderr, "metrics: empty source %s\n", spec);
    munmap(base, st.st_size); close(fd); return 0;
  }
  src->rows = malloc(n * ROW_COLS * sizeof(int32_t));
  if (!src->rows) { munmap(base, st.st_size); close(fd); return 0; }
  src->n = 0;
  for (long i = 0; i < n; i++) {
    long r = start + i * step;
    if (r >= hi) break;
    memcpy(src->rows + i * ROW_COLS, base + r * ROW_COLS, ROW_COLS * sizeof(int32_t));
    src->n++;
  }
  munmap(base, st.st_size); close(fd);
  snprintf(src->label, sizeof src->label, "%s", spec);
  return 1;
}

/* ── reductions ────────────────────────────────────────────────────────────── */

static double mean_d(const double *x, long n) {
  double s = 0; for (long i = 0; i < n; i++) s += x[i]; return n ? s / n : 0;
}
static double std_d(const double *x, long n) {
  double m = mean_d(x, n), s2 = 0;
  for (long i = 0; i < n; i++) { double d = x[i] - m; s2 += d * d; }
  return n ? sqrt(s2 / n) : 0;
}
static double mean_i(const int32_t *x, long n) {
  double s = 0; for (long i = 0; i < n; i++) s += x[i]; return n ? s / n : 0;
}
static double std_i(const int32_t *x, long n) {
  double m = mean_i(x, n), s2 = 0;
  for (long i = 0; i < n; i++) { double d = x[i] - m; s2 += d * d; }
  return n ? sqrt(s2 / n) : 0;
}

static int cmp_i32(const void *a, const void *b) {
  int32_t x = *(int32_t*)a, y = *(int32_t*)b;
  return (x > y) - (x < y);
}
static double median_i(const int32_t *x, long n) {
  if (n == 0) return 0;
  int32_t *t = malloc(n * sizeof *t);
  memcpy(t, x, n * sizeof *t);
  qsort(t, n, sizeof *t, cmp_i32);
  double m = (n & 1) ? (double)t[n/2] : 0.5 * ((double)t[n/2 - 1] + t[n/2]);
  free(t); return m;
}

/* ── column builders ───────────────────────────────────────────────────────── */

static int32_t *col_spread(struct Source *s) {
  int32_t *r = malloc(s->n * sizeof *r);
  for (long i = 0; i < s->n; i++) r[i] = AR(s, i) - BR(s, i);
  return r;
}
static int32_t *col_mid2(struct Source *s) {
  int32_t *r = malloc(s->n * sizeof *r);
  for (long i = 0; i < s->n; i++) r[i] = AR(s, i) + BR(s, i);
  return r;
}
static int32_t *col_nc0(struct Source *s) {
  int32_t *r = malloc(s->n * sizeof *r);
  for (long i = 0; i < s->n; i++) r[i] = AN(s, i);
  return r;
}
static int32_t *col_nc1(struct Source *s) {
  int32_t *r = malloc(s->n * sizeof *r);
  for (long i = 0; i < s->n; i++) r[i] = AN1(s, i);
  return r;
}
static double *returns1_from_mid2(const int32_t *m2, long n) {
  if (n < 2) return NULL;
  double *r = malloc((n - 1) * sizeof *r);
  for (long i = 0; i < n - 1; i++) r[i] = (double)(m2[i+1] - m2[i]);
  return r;
}
static double *imb0_arr(struct Source *s) {
  double *r = malloc(s->n * sizeof *r);
  for (long i = 0; i < s->n; i++) {
    double a = AN(s, i), b = BN(s, i), t = a + b;
    r[i] = t > 0 ? (a - b) / t : 0;
  }
  return r;
}

/* events_per_pair: merge-walk on one side counting elementary events.
 * Mirrors compare.py._walk_side_count. */
static long walk_side(const int32_t *pR, const int32_t *pN,
                      const int32_t *cR, const int32_t *cN, int diff) {
  int i = 0, j = 0; long n = 0;
  while (i < NL && j < NL && pN[i] != 0 && cN[j] != 0) {
    long d = (long)diff * ((long)cR[j] - pR[i]);
    if      (d < 0) { n += cN[j]; j++; }
    else if (d == 0) {
      long dn = (long)cN[j] - pN[i];
      if (dn) n += (dn < 0 ? -dn : dn);
      i++; j++;
    } else           { n += pN[i]; i++; }
  }
  return n;
}
static double events_per_pair(struct Source *s) {
  if (s->n < 2) return 0;
  long n = s->n - 1, total = 0;
  for (long t = 0; t < n; t++) {
    const int32_t *p = s->rows + t * ROW_COLS;
    const int32_t *c = s->rows + (t + 1) * ROW_COLS;
    total += walk_side(p + 0, p + 32, c + 0, c + 32, +1);   /* ask */
    total += walk_side(p + 8, p + 40, c + 8, c + 40, -1);   /* bid */
  }
  return (double)total / n;
}

/* ── sections ──────────────────────────────────────────────────────────────── */

static void hdr(const char *title) {
  printf("\n  %s\n", title);
}
static void row2(const char *label, double a, double b, const char *fmt) {
  char fa[32], fb[32];
  snprintf(fa, sizeof fa, "%%%d%s", W, fmt);
  snprintf(fb, sizeof fb, "%%%d%s", W, fmt);
  printf("  %-22s ", label);
  printf(fa, a); printf("  "); printf(fb, b);
  if (a != 0) printf("  %8.4f\n", b / a); else printf("\n");
}
static void row2_nodiff(const char *label, double a, double b, const char *fmt) {
  char fa[32]; snprintf(fa, sizeof fa, "%%%d%s", W, fmt);
  printf("  %-22s ", label);
  printf(fa, a); printf("  "); printf(fa, b); printf("\n");
}
static void dist_row(int key, double a, double b) {
  printf("  %4d   %*.2f  %*.2f  %+8.2f\n", key, W, a, W, b, b - a);
}
static void sec_flags(struct Source *A, struct Source *B) {
  hdr("0. FLAGS  (sim-vs-real gaps — >5% on key metrics)");
  double spA, spB, nA, nB, n1A, n1B, rA, rB;
  int32_t *spa = col_spread(A), *spb = col_spread(B);
  int32_t *m2a = col_mid2(A), *m2b = col_mid2(B);
  double *ra = returns1_from_mid2(m2a, A->n), *rb = returns1_from_mid2(m2b, B->n);
  spA = mean_i(spa, A->n); spB = mean_i(spb, B->n);
  nA = mean_i(&AN(A, 0), A->n) ? 0 : 0;   /* placeholder — use explicit arr */
  int32_t *na = col_nc0(A), *nb = col_nc0(B);
  int32_t *n1a = col_nc1(A), *n1b = col_nc1(B);
  nA = mean_i(na, A->n); nB = mean_i(nb, B->n);
  n1A = mean_i(n1a, A->n); n1B = mean_i(n1b, B->n);
  rA = std_d(ra, A->n - 1); rB = std_d(rb, B->n - 1);
  double evA = events_per_pair(A), evB = events_per_pair(B);
  int flagged = 0;
  if (evA > 0 && fabs(evB - evA) / evA > 0.05) {
    char lbl[64]; snprintf(lbl, sizeof lbl, "events/pair %+.1f%%", 100 * (evB/evA - 1));
    row2(lbl, evA, evB, ".4f"); flagged++;
  }
  if (spA > 0 && fabs(spB - spA) / spA > 0.05) {
    char lbl[64]; snprintf(lbl, sizeof lbl, "spread mean %+.1f%%", 100 * (spB/spA - 1));
    row2(lbl, spA, spB, ".3f"); flagged++;
  }
  if (nA > 0 && fabs(nB - nA) / nA > 0.05) {
    char lbl[64]; snprintf(lbl, sizeof lbl, "nc0 mean %+.1f%%", 100 * (nB/nA - 1));
    row2(lbl, nA, nB, ".3f"); flagged++;
  }
  if (n1A > 0 && fabs(n1B - n1A) / n1A > 0.05) {
    char lbl[64]; snprintf(lbl, sizeof lbl, "nc1 mean %+.1f%%", 100 * (n1B/n1A - 1));
    row2(lbl, n1A, n1B, ".3f"); flagged++;
  }
  if (rA > 0 && fabs(rB - rA) / rA > 0.05) {
    char lbl[64]; snprintf(lbl, sizeof lbl, "return std %+.1f%%", 100 * (rB/rA - 1));
    row2(lbl, rA, rB, ".3f"); flagged++;
  }
  if (!flagged) printf("  no gross discrepancies (<5%% on ev/pair, spread, nc0, nc1, ret std)\n");
  free(spa); free(spb); free(m2a); free(m2b); free(ra); free(rb);
  free(na); free(nb); free(n1a); free(n1b);
}

static void sec_basics(struct Source *A, struct Source *B) {
  hdr("1. BASICS");
  int32_t *spa = col_spread(A), *spb = col_spread(B);
  int32_t *na = col_nc0(A), *nb = col_nc0(B);
  int32_t *n1a = col_nc1(A), *n1b = col_nc1(B);
  row2_nodiff("ticks",         (double)A->n,      (double)B->n,      ",.0f");
  double evA = events_per_pair(A), evB = events_per_pair(B);
  row2("events/pair",          evA,               evB,               ".4f");
  row2("spread mean",          mean_i(spa, A->n), mean_i(spb, B->n), ".4f");
  row2("spread std",           std_i(spa, A->n),  std_i(spb, B->n),  ".4f");
  row2_nodiff("spread median", median_i(spa, A->n), median_i(spb, B->n), ".1f");
  row2("nc0 mean",             mean_i(na, A->n),  mean_i(nb, B->n),  ".4f");
  row2("nc0 std",              std_i(na, A->n),   std_i(nb, B->n),   ".4f");
  row2("nc1 mean",             mean_i(n1a, A->n), mean_i(n1b, B->n), ".4f");
  row2("nc1 std",              std_i(n1a, A->n),  std_i(n1b, B->n),  ".4f");
  free(spa); free(spb); free(na); free(nb); free(n1a); free(n1b);
}

/* Spread distribution: % of ticks by sp0 value, filter for ≥0.5% in either. */
static void sec_spread_dist(struct Source *A, struct Source *B) {
  hdr("2. SPREAD DISTRIBUTION  (% of ticks)  sp0");
  int32_t *spa = col_spread(A), *spb = col_spread(B);
  int32_t mn = INT32_MAX, mx = INT32_MIN;
  for (long i = 0; i < A->n; i++) {
    if (spa[i] < mn) mn = spa[i]; if (spa[i] > mx) mx = spa[i];
  }
  for (long i = 0; i < B->n; i++) {
    if (spb[i] < mn) mn = spb[i]; if (spb[i] > mx) mx = spb[i];
  }
  if (mn < 0) mn = 0;
  if (mx > 200) mx = 200;
  long *ca = calloc(mx - mn + 1, sizeof *ca), *cb = calloc(mx - mn + 1, sizeof *cb);
  for (long i = 0; i < A->n; i++) if (spa[i] >= mn && spa[i] <= mx) ca[spa[i] - mn]++;
  for (long i = 0; i < B->n; i++) if (spb[i] >= mn && spb[i] <= mx) cb[spb[i] - mn]++;
  for (int32_t v = mn; v <= mx; v++) {
    double pa = 100.0 * ca[v - mn] / A->n, pb = 100.0 * cb[v - mn] / B->n;
    if (pa < 0.5 && pb < 0.5) continue;
    dist_row(v, pa, pb);
  }
  free(ca); free(cb); free(spa); free(spb);
}

static void sec_nc0_dist(struct Source *A, struct Source *B) {
  hdr("3. NC0 DISTRIBUTION  (% of ticks, ask side)  nc0");
  long ca[11] = {0}, cb[11] = {0};
  for (long i = 0; i < A->n; i++) {
    int32_t v = AN(A, i); if (v > 10) v = 10; if (v >= 0) ca[v]++;
  }
  for (long i = 0; i < B->n; i++) {
    int32_t v = AN(B, i); if (v > 10) v = 10; if (v >= 0) cb[v]++;
  }
  for (int v = 1; v <= 10; v++) {
    double pa = 100.0 * ca[v] / A->n, pb = 100.0 * cb[v] / B->n;
    if (pa < 0.5 && pb < 0.5) continue;
    dist_row(v, pa, pb);
  }
}

/* Skewness and excess kurtosis (scipy "fisher" default: kurt - 3). */
static void moments(const double *x, long n, double *m, double *s,
                    double *sk, double *ku) {
  *m = mean_d(x, n);
  double s2 = 0, s3 = 0, s4 = 0;
  for (long i = 0; i < n; i++) {
    double d = x[i] - *m, d2 = d * d;
    s2 += d2; s3 += d2 * d; s4 += d2 * d2;
  }
  *s = n ? sqrt(s2 / n) : 0;
  double var = *s * *s;
  *sk = (n && var > 0) ? (s3 / n) / (var * *s) : 0;
  *ku = (n && var > 0) ? (s4 / n) / (var * var) - 3.0 : 0;
}

static void sec_return_dist(struct Source *A, struct Source *B) {
  hdr("5. PRICE-RETURN DISTRIBUTION  (lag-1, half-ticks)");
  int32_t *m2a = col_mid2(A), *m2b = col_mid2(B);
  double *ra = returns1_from_mid2(m2a, A->n), *rb = returns1_from_mid2(m2b, B->n);
  double ma, sa, ska, kua, mb, sb, skb, kub;
  moments(ra, A->n - 1, &ma, &sa, &ska, &kua);
  moments(rb, B->n - 1, &mb, &sb, &skb, &kub);
  row2("mean",          ma,  mb,  ".4f");
  row2("std",           sa,  sb,  ".4f");
  row2_nodiff("skewness", ska, skb, ".4f");
  row2_nodiff("exc. kurtosis", kua, kub, ".4f");
  free(m2a); free(m2b); free(ra); free(rb);
}

/* ACF at lags 1, 2, 3, 5, 10, 20, 50. */
static void sec_return_acf(struct Source *A, struct Source *B) {
  hdr("6. PRICE-RETURN AUTOCORRELATION  lag");
  const int lags[] = {1, 2, 3, 5, 10, 20, 50};
  const int K = sizeof lags / sizeof *lags;
  int32_t *m2a = col_mid2(A), *m2b = col_mid2(B);
  double *ra = returns1_from_mid2(m2a, A->n), *rb = returns1_from_mid2(m2b, B->n);
  long na = A->n - 1, nb = B->n - 1;
  double mua = mean_d(ra, na), mub = mean_d(rb, nb);
  double va = 0, vb = 0;
  for (long i = 0; i < na; i++) { double d = ra[i] - mua; va += d * d; }
  for (long i = 0; i < nb; i++) { double d = rb[i] - mub; vb += d * d; }
  va /= (na > 0 ? na : 1); vb /= (nb > 0 ? nb : 1);
  for (int k = 0; k < K; k++) {
    int L = lags[k];
    double aca = 0, acb = 0;
    if (va > 0 && na > L + 10) {
      for (long i = 0; i < na - L; i++) aca += (ra[i] - mua) * (ra[i + L] - mua);
      aca /= (na - L); aca /= va;
    }
    if (vb > 0 && nb > L + 10) {
      for (long i = 0; i < nb - L; i++) acb += (rb[i] - mub) * (rb[i + L] - mub);
      acb /= (nb - L); acb /= vb;
    }
    printf("  %4d   %*.4f  %*.4f\n", L, W, aca, W, acb);
  }
  free(m2a); free(m2b); free(ra); free(rb);
}

/* Mid-price prediction: d = B.mid - A.mid per row, y = A.rows[:, 48] / 4.
 * Report corr(d, y), R² = corr², and R² from OLS fit y ≈ β·d. */
static void sec_price_prediction(struct Source *A, struct Source *B) {
  hdr("7. MID-PRICE PREDICTION  (d = B.mid − A.mid, y = A.y/4)");
  long n = A->n < B->n ? A->n : B->n;
  if (n == 0) { printf("  (no pairs)\n"); return; }
  double *d = malloc(n * sizeof *d), *y = malloc(n * sizeof *y);
  for (long i = 0; i < n; i++) {
    d[i] = 0.5 * (AR(B, i) + BR(B, i)) - 0.5 * (AR(A, i) + BR(A, i));
    y[i] = (double)Y(A, i) / 4.0;
  }
  double md = mean_d(d, n), my = mean_d(y, n);
  double sdd = 0, syy = 0, sdy = 0;
  for (long i = 0; i < n; i++) {
    double dc = d[i] - md, yc = y[i] - my;
    sdd += dc * dc; syy += yc * yc; sdy += dc * yc;
  }
  double corr = 0, r2_fit = 0;
  if (sdd > 0 && syy > 0) corr = sdy / sqrt(sdd * syy);
  if (sdd > 0 && syy > 0) {
    double beta = sdy / sdd, ssres = 0;
    for (long i = 0; i < n; i++) { double e = y[i] - beta * d[i]; ssres += e * e; }
    r2_fit = 1.0 - ssres / syy;
  }
  long npos = 0, nneg = 0, nzero = 0;
  for (long i = 0; i < n; i++) {
    if      (d[i] > 0) npos++;
    else if (d[i] < 0) nneg++;
    else               nzero++;
  }
  double d_std = std_d(d, n), abs_m = 0;
  for (long i = 0; i < n; i++) abs_m += fabs(d[i]);
  abs_m /= n;
  printf("  %-22s %*ld\n", "pairs", W, n);
  printf("  %-22s %*.4f\n", "dmid mean", W, md);
  printf("  %-22s %*.4f\n", "dmid std", W, d_std);
  printf("  %-22s %*.4f\n", "|dmid| mean", W, abs_m);
  printf("  %-22s %*.2f\n", "dmid > 0 %", W, 100.0 * npos / n);
  printf("  %-22s %*.2f\n", "dmid = 0 %", W, 100.0 * nzero / n);
  printf("  %-22s %*.2f\n", "dmid < 0 %", W, 100.0 * nneg / n);
  printf("  %-22s %*.4f\n", "corr(dmid, y)", W, corr);
  printf("  %-22s %*.4f\n", "R^2 = corr^2", W, corr * corr);
  printf("  %-22s %*.4f\n", "R^2 (best-fit)", W, r2_fit);
  free(d); free(y);
}

static void sec_queue_imbalance(struct Source *A, struct Source *B) {
  hdr("13. QUEUE IMBALANCE  imb = (aN0 - bN0) / (aN0 + bN0)");
  double *ia = imb0_arr(A), *ib = imb0_arr(B);
  double ma = mean_d(ia, A->n), mb = mean_d(ib, B->n);
  double sa = std_d(ia, A->n), sb = std_d(ib, B->n);
  double abs_a = 0, abs_b = 0;
  long aha = 0, ahb = 0, bha = 0, bhb = 0, bala = 0, balb = 0;
  for (long i = 0; i < A->n; i++) {
    abs_a += fabs(ia[i]);
    if (ia[i] >  0.2) aha++; else if (ia[i] < -0.2) bha++;
    if (fabs(ia[i]) < 0.1) bala++;
  }
  for (long i = 0; i < B->n; i++) {
    abs_b += fabs(ib[i]);
    if (ib[i] >  0.2) ahb++; else if (ib[i] < -0.2) bhb++;
    if (fabs(ib[i]) < 0.1) balb++;
  }
  abs_a /= A->n; abs_b /= B->n;
  row2("mean",               ma, mb, ".4f");
  row2("std",                sa, sb, ".4f");
  row2("|mean|",             abs_a, abs_b, ".4f");
  row2("ask-heavy% (>0.2)",  100.0*aha/A->n, 100.0*ahb/B->n, ".2f");
  row2("bid-heavy% (<-0.2)", 100.0*bha/A->n, 100.0*bhb/B->n, ".2f");
  row2("balanced% (|i|<.1)", 100.0*bala/A->n, 100.0*balb/B->n, ".2f");
  free(ia); free(ib);
}

static const double IMB_EDGES[6]  = {-1.001, -0.3, -0.1, 0.1, 0.3, 1.001};
static const char * const IMB_LBL[5] = {"<-.3","-.3..-.1","-.1..+.1","+.1..+.3",">+.3"};

static int imb_bin(double x) {
  for (int i = 0; i < 5; i++) if (x <= IMB_EDGES[i+1]) return i;
  return 4;
}

static void sec_drift_by_imb(struct Source *A, struct Source *B) {
  hdr("14. E[Δmid | imb_bin]  (directional drift per state)  imb");
  int32_t *m2a = col_mid2(A), *m2b = col_mid2(B);
  double *ia = imb0_arr(A), *ib = imb0_arr(B);
  double sa[5] = {0}, sb[5] = {0}; long ca[5] = {0}, cb[5] = {0};
  for (long i = 0; i < A->n - 1; i++) {
    int bn = imb_bin(ia[i]);
    sa[bn] += (double)(m2a[i+1] - m2a[i]); ca[bn]++;
  }
  for (long i = 0; i < B->n - 1; i++) {
    int bn = imb_bin(ib[i]);
    sb[bn] += (double)(m2b[i+1] - m2b[i]); cb[bn]++;
  }
  for (int i = 0; i < 5; i++) {
    double a = ca[i] > 50 ? sa[i] / ca[i] : NAN;
    double b = cb[i] > 50 ? sb[i] / cb[i] : NAN;
    printf("  %-10s ", IMB_LBL[i]);
    if (isnan(a)) printf("%*s", W, "n/a"); else printf("%+*.4f", W, a);
    printf("  ");
    if (isnan(b)) printf("%*s", W, "n/a"); else printf("%+*.4f", W, b);
    if (!isnan(a) && !isnan(b)) printf("  %+8.4f\n", b - a);
    else printf("\n");
  }
  free(m2a); free(m2b); free(ia); free(ib);
}

/* Classify each adjacent (t, t+1) pair into a 5-way event type:
 *   0 none,  1 ask_add,  2 ask_rem,  3 bid_add,  4 bid_rem
 * Matches compare.py §25 — catches which side/direction fired each tick. */
static int classify_event(struct Source *s, long t) {
  int32_t aR0 = AR(s, t),   bR0 = BR(s, t);
  int32_t aR1 = AR(s, t+1), bR1 = BR(s, t+1);
  int32_t aN0 = AN(s, t),   bN0 = BN(s, t);
  int32_t aN1 = AN(s, t+1), bN1 = BN(s, t+1);
  if (aR1 == aR0 && aN1 > aN0) return 1;        /* ask_add (queue grew) */
  if (aR1 != aR0 || aN1 < aN0) return 2;        /* ask_rem or move     */
  if (bR1 == bR0 && bN1 > bN0) return 3;        /* bid_add */
  if (bR1 != bR0 || bN1 < bN0) return 4;        /* bid_rem */
  return 0;                                      /* no change */
}

static void sec_event_transitions(struct Source *A, struct Source *B) {
  hdr("25. EVENT-TYPE TRANSITION  P(next | last) (%)");
  const char *labels[5] = {"none", "a+", "a-", "b+", "b-"};
  long cntA[5][5] = {{0}}, cntB[5][5] = {{0}};
  if (A->n >= 3) {
    int prev = classify_event(A, 0);
    for (long t = 1; t < A->n - 1; t++) {
      int cur = classify_event(A, t);
      cntA[prev][cur]++; prev = cur;
    }
  }
  if (B->n >= 3) {
    int prev = classify_event(B, 0);
    for (long t = 1; t < B->n - 1; t++) {
      int cur = classify_event(B, t);
      cntB[prev][cur]++; prev = cur;
    }
  }
  printf("     %-8s  ", "from→");
  for (int j = 0; j < 5; j++) printf(" %5s ", labels[j]);
  printf("\n");
  for (int i = 0; i < 5; i++) {
    long sA = 0, sB = 0;
    for (int j = 0; j < 5; j++) { sA += cntA[i][j]; sB += cntB[i][j]; }
    if (sA < 50 && sB < 50) continue;
    printf("  A  %-6s : ", labels[i]);
    for (int j = 0; j < 5; j++)
      printf(" %5.1f ", sA > 0 ? 100.0 * cntA[i][j] / sA : 0);
    printf("\n  B  %-6s : ", labels[i]);
    for (int j = 0; j < 5; j++)
      printf(" %5.1f ", sB > 0 ? 100.0 * cntB[i][j] / sB : 0);
    printf("\n");
  }
}

/* Price impact: E[Δmid(t+k) | ask-removal at t].  Signals how sim vs real
 * respond to a hit on the ask side.  Asymmetric impact over k reveals
 * order-flow toxicity / momentum structure. */
static void sec_price_impact(struct Source *A, struct Source *B) {
  hdr("29. PRICE IMPACT  E[Δmid(t+k) | ask-event at t] (half-ticks)  k");
  const int ks[] = {1, 2, 5, 10, 20};
  const int NK = sizeof ks / sizeof *ks;
  for (int ki = 0; ki < NK; ki++) {
    int k = ks[ki];
    double impA = NAN, impB = NAN;
    long cntA = 0, cntB = 0;
    for (int which = 0; which < 2; which++) {
      struct Source *s = which ? B : A;
      if (s->n < k + 2) continue;
      double sum = 0; long n = 0;
      for (long t = 0; t < s->n - k - 1; t++) {
        int32_t aR0 = AR(s, t),   aR1 = AR(s, t+1);
        int32_t aN0 = AN(s, t),   aN1 = AN(s, t+1);
        int ask_rem = (aR1 > aR0) || (aR1 == aR0 && aN1 < aN0);
        if (!ask_rem) continue;
        double m_t  = 0.5 * (AR(s, t+1)   + BR(s, t+1));
        double m_tk = 0.5 * (AR(s, t+1+k) + BR(s, t+1+k));
        sum += m_tk - m_t; n++;
      }
      if (which) { cntB = n; if (n > 50) impB = sum / n; }
      else       { cntA = n; if (n > 50) impA = sum / n; }
    }
    printf("  %4d  ", k);
    if (!isnan(impA)) printf("%+*.4f", W, impA); else printf("%*s", W, "n/a");
    printf("  ");
    if (!isnan(impB)) printf("%+*.4f", W, impB); else printf("%*s", W, "n/a");
    if (!isnan(impA) && !isnan(impB)) printf("  %+8.4f", impB - impA);
    printf("  (nA=%ld nB=%ld)\n", cntA, cntB);
  }
}

/* ── main ──────────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: metrics <specA> <specB>\n");
    fprintf(stderr, "  spec = path[:tag[:stride]]   tag = int|odd|even\n");
    return 1;
  }
  struct Source A = {0}, B = {0};
  if (!load_source(argv[1], &A)) return 1;
  if (!load_source(argv[2], &B)) return 1;
  printf("\n======================================================================\n");
  printf("  A: %s  (n=%ld)\n", A.label, A.n);
  printf("  B: %s  (n=%ld)\n", B.label, B.n);
  printf("======================================================================\n");
  printf("  %-22s %*s  %*s  %8s\n", "", W, "A", W, "B", "B/A");

  sec_flags(&A, &B);
  sec_basics(&A, &B);
  sec_spread_dist(&A, &B);
  sec_nc0_dist(&A, &B);
  sec_return_dist(&A, &B);
  sec_return_acf(&A, &B);
  sec_price_prediction(&A, &B);
  sec_queue_imbalance(&A, &B);
  sec_drift_by_imb(&A, &B);
  sec_event_transitions(&A, &B);
  sec_price_impact(&A, &B);

  printf("\n======================================================================\n\n");
  free(A.rows); free(B.rows);
  return 0;
}
