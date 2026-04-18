#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { nl = 8 };
struct Row {
  int32_t aR[nl], bR[nl], aS[nl], bS[nl], aN[nl], bN[nl], y;
};
struct Counter {
  long long tp, tm, dp, dm, r;
  long long *tbl[2][2];
};
static void counter_init(struct Counter *c) {
  c->tp = c->tm = c->dp = c->dm = c->r = 0;
  c->tbl[0][0] = &c->tp;
  c->tbl[0][1] = &c->tm;
  c->tbl[1][0] = &c->dp;
  c->tbl[1][1] = &c->dm;
}
static int diff_ask(int32_t a, int32_t b) { return a - b; }
static int diff_bid(int32_t a, int32_t b) { return b - a; }
static void walk(int32_t *pR, int32_t *pN, int32_t *pS, int32_t *cR,
                 int32_t *cN, int32_t *cS, int (*diff)(int32_t, int32_t),
                 struct Counter *c) {
  int i = 0, j = 0;
  int32_t dn, ds, d;
  while (i < nl && j < nl && pN[i] != 0 && cN[j] != 0) {
    d = diff(cR[j], pR[i]);
    if (d < 0) {
      (*c->tbl[j != 0][0]) += cN[j];
      j++;
    } else if (d == 0) {
      dn = cN[j] - pN[i];
      ds = cS[j] - pS[i];
      if (dn)
        (*c->tbl[j != 0][dn < 0]) += dn < 0 ? -dn : dn;
      else if (ds != 0)
        c->r++;
      i++;
      j++;
    } else {
      (*c->tbl[i != 0][1]) += pN[i];
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
  struct Counter ask, bid;
  long long ntics = 0, n = 0;
  counter_init(&ask);
  counter_init(&bid);
  while (fread(&prev, sizeof prev, 1, stdin) == 1 &&
         fread(&cur, sizeof cur, 1, stdin) == 1) {
    ntics++;
    if (memcmp(&prev, &cur, sizeof prev) == 0) {
      n++;
      continue;
    }
    walk(prev.aR, prev.aN, prev.aS, cur.aR, cur.aN, cur.aS, diff_ask, &ask);
    walk(prev.bR, prev.bN, prev.bS, cur.bR, cur.bN, cur.bS, diff_bid, &bid);
  }
  long long out[] = {ntics, n, ask.tp, ask.tm, ask.dp, ask.dm, ask.r,
                     bid.tp, bid.tm, bid.dp, bid.dm, bid.r};
  size_t i;
  for (i = 0; i < sizeof out / sizeof *out; i++) {
    if (i)
      putchar(' ');
    printf("%10lld", out[i]);
  }
  putchar('\n');
}
