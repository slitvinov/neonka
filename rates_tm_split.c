/* rates_tm_split — like rates.c, but tm is split into tm_q (pre-event N0>1,
 * queue decrement) and tm_c (pre-event N0=1, cascade).  This calibration
 * matters for sim stability: at sp=40, tm_c is only ~26% of total tm but
 * our current code uses tm_total as the cascade rate, overfiring cascades
 * 4× and driving sp drift.
 *
 * Output per line (bucketed by sp or (sp, imb)):
 *   key  ntics  n  a_tp a_tmq a_tmc a_dp a_dm a_r  b_tp b_tmq b_tmc b_dp b_dm b_r  ...
 *
 * Columns 2-13 are event counts; 14+ are summary fields (preserved from rates.c).
 *
 * Usage: ./rates_tm_split -B sp0_imb < pairs
 */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { nl = 8, SPMAX = 2000, IMB_BINS = 6 };

static int imb_bin(int32_t aN0, int32_t bN0, int32_t aN1, int32_t bN1) {
  int64_t s = (int64_t)aN0 + bN0, d = (int64_t)aN0 - bN0;
  int b0 = (s == 0) ? 1 : (d * 5 < -s) ? 0 : (d * 5 > s) ? 2 : 1;
  int s1 = (aN1 > bN1) ? 1 : 0;
  return b0 * 2 + s1;
}

struct Row {
  int32_t aR[nl], bR[nl], aS[nl], bS[nl], aN[nl], bN[nl], y;
};

struct Bucket {
  long long ntics, n;
  long long a_tp, a_tmq, a_tmc, a_dp, a_dm, a_r;
  long long b_tp, b_tmq, b_tmc, b_dp, b_dm, b_r;
  long long sum_aN0, sum_bN0, sum_aNd, sum_bNd;
};

static int diff_ask(int32_t a, int32_t b) { return a - b; }
static int diff_bid(int32_t a, int32_t b) { return b - a; }

/* tm at level 0 splits by pre-event N[0]: pN[0] > 1 → tmq, pN[0] == 1 → tmc. */
static void walk(int32_t *pR, int32_t *pN, int32_t *pS, int32_t *cR,
                 int32_t *cN, int32_t *cS, int (*diff)(int32_t, int32_t),
                 long long *tp, long long *tmq, long long *tmc,
                 long long *dp, long long *dm, long long *r) {
  int i = 0, j = 0;
  int32_t dn, ds, d;
  while (i < nl && j < nl && pN[i] != 0 && cN[j] != 0) {
    d = diff(cR[j], pR[i]);
    if (d < 0) {
      if (j == 0) *tp += cN[j];                        /* new top level */
      else        *dp += cN[j];                        /* deeper insertion */
      j++;
    } else if (d == 0) {
      dn = cN[j] - pN[i];
      ds = cS[j] - pS[i];
      if (dn > 0) {
        if (j == 0) *tp += dn;
        else        *dp += dn;
      } else if (dn < 0) {
        if (j == 0) {
          /* Split tm at top: cascade (pN[0]=1 with full removal) vs queue. */
          if (pN[i] == 1 && cN[j] == 0) *tmc += 1;
          else                           *tmq += -dn;
        } else *dm += -dn;
      } else if (ds != 0) (*r)++;
      i++; j++;
    } else {
      /* Prev level disappeared — at top this is tm. Pre-event N[0] = pN[i]. */
      if (i == 0) {
        if (pN[i] == 1) *tmc += 1;
        else            *tmq += pN[i];
      } else *dm += pN[i];
      i++;
    }
  }
}

static void emit(int32_t key, struct Bucket *b, int with_key) {
  long long out[] = {b->ntics, b->n,
                     b->a_tp, b->a_tmq, b->a_tmc, b->a_dp, b->a_dm, b->a_r,
                     b->b_tp, b->b_tmq, b->b_tmc, b->b_dp, b->b_dm, b->b_r,
                     b->sum_aN0, b->sum_bN0, b->sum_aNd, b->sum_bNd};
  if (with_key) printf("%6d ", key);
  for (size_t i = 0; i < sizeof out / sizeof *out; i++) {
    if (i) putchar(' ');
    printf("%10lld", out[i]);
  }
  putchar('\n');
}

int main(int argc, char **argv) {
  const char *bucket = "sp0_imb";
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-B") && i + 1 < argc) bucket = argv[++i];
    else { fprintf(stderr, "usage: rates_tm_split -B sp0_imb\n"); return 1; }
  }

  static struct Bucket bins[SPMAX * IMB_BINS];
  memset(bins, 0, sizeof bins);

  struct Row prev, cur;
  while (fread(&prev, sizeof prev, 1, stdin) == 1 &&
         fread(&cur,  sizeof cur,  1, stdin) == 1) {
    int32_t sp = prev.aR[0] - prev.bR[0];
    if (sp < 0 || sp >= SPMAX) continue;
    int im = imb_bin(prev.aN[0], prev.bN[0], prev.aN[1], prev.bN[1]);
    struct Bucket *b = &bins[sp * IMB_BINS + im];
    b->ntics++;
    if (memcmp(&prev, &cur, sizeof prev) == 0) { b->n++; continue; }
    walk(prev.aR, prev.aN, prev.aS, cur.aR, cur.aN, cur.aS, diff_ask,
         &b->a_tp, &b->a_tmq, &b->a_tmc, &b->a_dp, &b->a_dm, &b->a_r);
    walk(prev.bR, prev.bN, prev.bS, cur.bR, cur.bN, cur.bS, diff_bid,
         &b->b_tp, &b->b_tmq, &b->b_tmc, &b->b_dp, &b->b_dm, &b->b_r);
    b->sum_aN0 += prev.aN[0]; b->sum_bN0 += prev.bN[0];
  }

  if (!strcmp(bucket, "sp0_imb")) {
    for (int sp = 0; sp < SPMAX; sp++)
      for (int im = 0; im < IMB_BINS; im++) {
        struct Bucket *b = &bins[sp * IMB_BINS + im];
        if (b->ntics > 0) { printf("%6d %2d ", sp, im); emit(0, b, 0); }
      }
  } else {
    fprintf(stderr, "rates_tm_split: unknown bucket '%s'\n", bucket);
    return 1;
  }
  return 0;
}
