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

static void walk(struct Row *prev, struct Row *cur, int side,
                 int (*diff)(int32_t, int32_t), int64_t *tbl[2][2],
                 int64_t *r) {
  int32_t *pR = side ? prev->bR : prev->aR;
  int32_t *pN = side ? prev->bN : prev->aN;
  int32_t *pS = side ? prev->bS : prev->aS;
  int32_t *cR = side ? cur->bR : cur->aR;
  int32_t *cN = side ? cur->bN : cur->aN;
  int32_t *cS = side ? cur->bS : cur->aS;
  int i = 0, j = 0;
  int32_t dn, ds, d;
  while (i < nl && j < nl && pN[i] != 0 && cN[j] != 0) {
    d = diff(cR[j], pR[i]);
    if (d < 0) {
      (*tbl[j != 0][0])++;
      j++;
    } else if (d == 0) {
      dn = cN[j] - pN[i];
      ds = cS[j] - pS[i];
      if (dn)
        (*tbl[j != 0][dn < 0])++;
      else if (ds != 0)
        (*r)++;
      i++;
      j++;
    } else {
      (*tbl[i != 0][1])++;
      i++;
    }
  }
}

int main(int argc, char **argv) {
  int i_arg;
  for (i_arg = 1; i_arg < argc; i_arg++) {
    if (strcmp(argv[i_arg], "-H") == 0) {
      printf("%10s %10s %10s %10s %10s %10s %10s\n", "ntics", "tp", "tm", "dp",
             "dm", "r", "n");
      return 0;
    }
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
    walk(&prev, &cur, 0, diff_ask, tbl, &r);
    walk(&prev, &cur, 1, diff_bid, tbl, &r);
  }
  printf("%10lld %10lld %10lld %10lld %10lld %10lld %10lld\n", (long long)ntics,
         (long long)tp, (long long)tm, (long long)dp, (long long)dm,
         (long long)r, (long long)n);
  return 0;
}
