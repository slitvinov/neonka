#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { nl = 8, NB = 10000 };
struct Row {
  int32_t aR[nl], bR[nl], aS[nl], bS[nl], aN[nl], bN[nl], y;
};

static int diff_ask(int32_t a, int32_t b) { return a - b; }
static int diff_bid(int32_t a, int32_t b) { return b - a; }

static void walk(int32_t *pR, int32_t *pN, int32_t *cR, int32_t *cN,
                 int32_t best, int (*diff)(int32_t, int32_t), int64_t *bins) {
  int i = 0, j = 0;
  int32_t dn, d, dist;
  while (i < nl && j < nl && pN[i] != 0 && cN[j] != 0) {
    d = diff(cR[j], pR[i]);
    if (d < 0) {
      if (j > 0) {
        dist = diff(cR[j], best);
        if (dist >= 0 && dist < NB)
          bins[dist] += cN[j];
      }
      j++;
    } else if (d == 0) {
      dn = cN[j] - pN[i];
      if (j > 0 && dn > 0) {
        dist = diff(cR[j], best);
        if (dist >= 0 && dist < NB)
          bins[dist] += dn;
      }
      i++;
      j++;
    } else {
      i++;
    }
  }
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  struct Row prev, cur;
  int64_t *bins = calloc(NB, sizeof *bins);
  if (bins == NULL) {
    fprintf(stderr, "dp.c: error: calloc failed\n");
    return 1;
  }
  while (fread(&prev, sizeof prev, 1, stdin) == 1 &&
         fread(&cur, sizeof cur, 1, stdin) == 1) {
    walk(prev.aR, prev.aN, cur.aR, cur.aN, prev.aR[0], diff_ask, bins);
    walk(prev.bR, prev.bN, cur.bR, cur.bN, prev.bR[0], diff_bid, bins);
  }
  int i;
  for (i = 0; i < NB; i++)
    if (bins[i])
      printf("%d %lld\n", i, (long long)bins[i]);
  free(bins);
  return 0;
}
