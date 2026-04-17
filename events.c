#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { nl = 8 };
struct Row {
  int32_t aR[nl], bR[nl], aS[nl], bS[nl], aN[nl], bN[nl], y;
};

static int diff_ask(int32_t a, int32_t b) { return a - b; }
static int diff_bid(int32_t a, int32_t b) { return b - a; }

static void walk(int32_t *pR, int32_t *pN, int32_t *pS, int32_t *cR,
                 int32_t *cN, int32_t *cS, int (*diff)(int32_t, int32_t),
                 int64_t *tbl[2][2], int64_t *r) {
  int i = 0, j = 0;
  int32_t dn, ds, d;
  while (i < nl && j < nl && pN[i] != 0 && cN[j] != 0) {
    d = diff(cR[j], pR[i]);
    if (d < 0) {
      (*tbl[j != 0][0]) += pN[j];
      j++;
    } else if (d == 0) {
      dn = cN[j] - pN[i];
      ds = cS[j] - pS[i];
      if (dn)
        (*tbl[j != 0][dn < 0]) += dn < 0 ? -dn : dn;
      else if (ds != 0)
        (*r)++;
      i++;
      j++;
    } else {
      (*tbl[i != 0][1]) += pN[i];
      i++;
    }
  }
}

int main(int argc, char **argv) {
  int i_arg;
  for (i_arg = 1; i_arg < argc; i_arg++) {
    fprintf(stderr, "events.c: error: unknown flag '%s'\n", argv[i_arg]);
    return 1;
  }
  struct Row prev, cur;
  int64_t tp = 0, tm = 0, dp = 0, dm = 0, r = 0, n = 0, ntics = 0;
  int64_t *tbl[2][2] = {{&tp, &tm}, {&dp, &dm}};
  while (fread(&prev, sizeof prev, 1, stdin) == 1 &&
         fread(&cur, sizeof cur, 1, stdin) == 1) {
    ntics++;
    if (memcmp(&prev, &cur, sizeof prev) == 0) {
      n++;
      continue;
    }
    walk(prev.aR, prev.aN, prev.aS, cur.aR, cur.aN, cur.aS, diff_ask, tbl, &r);
    walk(prev.bR, prev.bN, prev.bS, cur.bR, cur.bN, cur.bS, diff_bid, tbl, &r);
  }
  printf("%10lld %10lld %10lld %10lld %10lld %10lld %10lld\n", (long long)ntics,
         (long long)tp, (long long)tm, (long long)dp, (long long)dm,
         (long long)r, (long long)n);
  return 0;
}
