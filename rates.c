#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { nl = 8, SPMAX = 2000, IMB_BINS = 6 };

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
};
struct Bucket {
  long long ntics, n;
  long long a_tp, a_tm, a_dp, a_dm, a_r;
  long long b_tp, b_tm, b_dp, b_dm, b_r;
  long long sum_aN0, sum_bN0, sum_aNd, sum_bNd;
};

static int diff_ask(int32_t a, int32_t b) { return a - b; }
static int diff_bid(int32_t a, int32_t b) { return b - a; }

static void walk(int32_t *pR, int32_t *pN, int32_t *pS, int32_t *cR,
                 int32_t *cN, int32_t *cS, int (*diff)(int32_t, int32_t),
                 long long *tp, long long *tm, long long *dp, long long *dm,
                 long long *r) {
  long long *tbl[2][2] = {{tp, tm}, {dp, dm}};
  int i = 0, j = 0;
  int32_t dn, ds, d;
  while (i < nl && j < nl && pN[i] != 0 && cN[j] != 0) {
    d = diff(cR[j], pR[i]);
    if (d < 0) {
      *tbl[j != 0][0] += cN[j];
      j++;
    } else if (d == 0) {
      dn = cN[j] - pN[i];
      ds = cS[j] - pS[i];
      if (dn)
        *tbl[j != 0][dn < 0] += dn < 0 ? -dn : dn;
      else if (ds != 0)
        (*r)++;
      i++;
      j++;
    } else {
      *tbl[i != 0][1] += pN[i];
      i++;
    }
  }
}

static void emit(int32_t key, struct Bucket *b, int with_key) {
  long long out[] = {b->ntics,  b->n,      b->a_tp,  b->a_tm, b->a_dp, b->a_dm,
                     b->a_r,    b->b_tp,   b->b_tm,  b->b_dp, b->b_dm, b->b_r,
                     b->sum_aN0, b->sum_bN0, b->sum_aNd, b->sum_bNd};
  size_t i;
  if (with_key)
    printf("%6d ", key);
  for (i = 0; i < sizeof out / sizeof *out; i++) {
    if (i)
      putchar(' ');
    printf("%10lld", out[i]);
  }
  putchar('\n');
}

int main(int argc, char **argv) {
  int bin_mode = 0, i;
  for (i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-B") && i + 1 < argc) {
      if (!strcmp(argv[i + 1], "sp0"))
        bin_mode = 1;
      else if (!strcmp(argv[i + 1], "sp0_imb"))
        bin_mode = 2;
      else {
        fprintf(stderr, "rates.c: error: unknown bin '%s'\n", argv[i + 1]);
        return 1;
      }
      i++;
      continue;
    }
    fprintf(stderr, "rates.c: error: unknown flag '%s'\n", argv[i]);
    return 1;
  }

  struct Row prev, cur;
  struct Bucket one = {0};
  struct Bucket *buckets = NULL;
  int nbuck = bin_mode == 2 ? SPMAX * IMB_BINS : bin_mode == 1 ? SPMAX : 0;
  if (nbuck) {
    buckets = calloc(nbuck, sizeof *buckets);
    if (buckets == NULL) {
      fprintf(stderr, "rates.c: error: calloc failed\n");
      return 1;
    }
  }

  while (fread(&prev, sizeof prev, 1, stdin) == 1 &&
         fread(&cur, sizeof cur, 1, stdin) == 1) {
    struct Bucket *b = &one;
    if (bin_mode) {
      int32_t sp = prev.aR[0] - prev.bR[0];
      if (sp < 0 || sp >= SPMAX)
        continue;
      int idx = bin_mode == 2
              ? sp * IMB_BINS + imb_bin(prev.aN[0], prev.bN[0],
                                        prev.aN[1], prev.bN[1])
              : sp;
      b = &buckets[idx];
    }
    b->ntics++;
    b->sum_aN0 += prev.aN[0];
    b->sum_bN0 += prev.bN[0];
    {
      int lv;
      long long sd_a = 0, sd_b = 0;
      for (lv = 1; lv < nl; lv++) {
        sd_a += prev.aN[lv];
        sd_b += prev.bN[lv];
      }
      b->sum_aNd += sd_a;
      b->sum_bNd += sd_b;
    }
    if (memcmp(&prev, &cur, sizeof prev) == 0) {
      b->n++;
      continue;
    }
    walk(prev.aR, prev.aN, prev.aS, cur.aR, cur.aN, cur.aS, diff_ask, &b->a_tp,
         &b->a_tm, &b->a_dp, &b->a_dm, &b->a_r);
    walk(prev.bR, prev.bN, prev.bS, cur.bR, cur.bN, cur.bS, diff_bid, &b->b_tp,
         &b->b_tm, &b->b_dp, &b->b_dm, &b->b_r);
  }

  if (bin_mode == 2) {
    int sp, im;
    for (sp = 0; sp < SPMAX; sp++)
      for (im = 0; im < IMB_BINS; im++) {
        struct Bucket *bk = &buckets[sp * IMB_BINS + im];
        if (bk->ntics > 0) {
          printf("%6d %2d ", sp, im);
          long long out[] = {bk->ntics, bk->n,   bk->a_tp, bk->a_tm,
                             bk->a_dp,  bk->a_dm, bk->a_r, bk->b_tp,
                             bk->b_tm,  bk->b_dp, bk->b_dm, bk->b_r,
                             bk->sum_aN0, bk->sum_bN0,
                             bk->sum_aNd, bk->sum_bNd};
          size_t j;
          for (j = 0; j < sizeof out / sizeof *out; j++) {
            if (j) putchar(' ');
            printf("%10lld", out[j]);
          }
          putchar('\n');
        }
      }
  } else if (bin_mode == 1) {
    for (i = 0; i < SPMAX; i++)
      if (buckets[i].ntics > 0)
        emit(i, &buckets[i], 1);
  } else {
    emit(0, &one, 0);
  }
  free(buckets);
  return 0;
}
