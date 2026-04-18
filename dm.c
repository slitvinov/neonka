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
                 int32_t best, int (*diff)(int32_t, int32_t), int64_t *expo,
                 int64_t *evt) {
  int k, i = 0, j = 0;
  int32_t dist, dn, d;
  for (k = 1; k < nl; k++) {
    if (pN[k] == 0)
      continue;
    dist = diff(pR[k], best);
    if (dist >= 0 && dist < NB)
      expo[dist] += pN[k];
  }
  while (i < nl && j < nl && pN[i] != 0 && cN[j] != 0) {
    d = diff(cR[j], pR[i]);
    if (d < 0) {
      j++;
    } else if (d == 0) {
      dn = cN[j] - pN[i];
      if (i > 0 && dn < 0) {
        dist = diff(pR[i], best);
        if (dist >= 0 && dist < NB)
          evt[dist] += -dn;
      }
      i++;
      j++;
    } else {
      if (i > 0) {
        dist = diff(pR[i], best);
        if (dist >= 0 && dist < NB)
          evt[dist] += pN[i];
      }
      i++;
    }
  }
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  struct Row prev, cur;
  int64_t *expo = calloc(NB, sizeof *expo);
  int64_t *evt = calloc(NB, sizeof *evt);
  if (expo == NULL || evt == NULL) {
    fprintf(stderr, "dm.c: error: calloc failed\n");
    return 1;
  }
  while (fread(&prev, sizeof prev, 1, stdin) == 1 &&
         fread(&cur, sizeof cur, 1, stdin) == 1) {
    walk(prev.aR, prev.aN, cur.aR, cur.aN, prev.aR[0], diff_ask, expo, evt);
    walk(prev.bR, prev.bN, cur.bR, cur.bN, prev.bR[0], diff_bid, expo, evt);
  }
  int i;
  for (i = 0; i < NB; i++)
    if (expo[i] || evt[i])
      printf("%d %lld %lld\n", i, (long long)expo[i], (long long)evt[i]);
  free(expo);
  free(evt);
  return 0;
}
