/* tp_sp — per-spread tp/dp jump distributions.
 *
 * Input  (stdin): paired Rows (prev, cur) — output of ./pairs.
 * Output (stdout): sp dist count  (one line per (sp, dist, count) triple)
 *
 * Mirrors tp.c/dp.c but buckets by pre-event spread sp = prev.aR[0] − prev.bR[0].
 * Pools ask + bid sides (by symmetry).  The tp_own/dp_own sampler in onestep
 * uses this to draw the jump distance for tp/dp events conditional on current
 * spread — essential for closing wide gaps in one step.
 *
 * Usage:
 *   ./tp_sp -e tp < pairs.bin    # tp jumps per sp
 *   ./tp_sp -e dp < pairs.bin    # dp jumps per sp
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { NL = 8, DIST_MAX = 1024, SP_MAX = 256 };
struct Row {
  int32_t aR[NL], bR[NL], aS[NL], bS[NL], aN[NL], bN[NL], y;
};

static int diff_ask(int32_t a, int32_t b) { return a - b; }
static int diff_bid(int32_t a, int32_t b) { return b - a; }

/* For a single side, bucket improvement distances (tp) or deep-level
 * insertions (dp). When ev==TP: j==0 emits; when ev==DP: j>0 emits. */
enum { EV_TP = 0, EV_DP = 1 };
static void walk(int32_t *pR, int32_t *pN, int32_t *cR, int32_t *cN,
                 int (*diff)(int32_t, int32_t), int ev,
                 int32_t sp, long long (*bins)[DIST_MAX]) {
  int i = 0, j = 0;
  int32_t d, dn, dist;
  while (i < NL && j < NL && pN[i] != 0 && cN[j] != 0) {
    d = diff(cR[j], pR[i]);
    if (d < 0) {
      dist = -d;
      if (dist > 0 && dist < DIST_MAX) {
        if (ev == EV_TP && j == 0)      bins[sp][dist] += cN[j];
        else if (ev == EV_DP && j > 0)  bins[sp][dist] += cN[j];
      }
      j++;
    } else if (d == 0) {
      dn = cN[j] - pN[i];
      if (dn > 0) {
        if (ev == EV_TP && j == 0)      bins[sp][0] += dn;
        else if (ev == EV_DP && j > 0)  bins[sp][0] += dn;
      }
      i++; j++;
    } else {
      i++;
    }
  }
}

int main(int argc, char **argv) {
  int ev = EV_TP;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-e") && i + 1 < argc) {
      const char *t = argv[++i];
      if      (!strcmp(t, "tp")) ev = EV_TP;
      else if (!strcmp(t, "dp")) ev = EV_DP;
      else { fprintf(stderr, "tp_sp: unknown event '%s'\n", t); return 1; }
    } else { fprintf(stderr, "usage: tp_sp -e tp|dp < pairs\n"); return 1; }
  }

  static long long bins[SP_MAX][DIST_MAX];
  memset(bins, 0, sizeof bins);

  struct Row prev, cur;
  while (fread(&prev, sizeof prev, 1, stdin) == 1 &&
         fread(&cur,  sizeof cur,  1, stdin) == 1) {
    int32_t sp = prev.aR[0] - prev.bR[0];
    if (sp < 0 || sp >= SP_MAX) continue;
    walk(prev.aR, prev.aN, cur.aR, cur.aN, diff_ask, ev, sp, bins);
    walk(prev.bR, prev.bN, cur.bR, cur.bN, diff_bid, ev, sp, bins);
  }

  for (int sp = 0; sp < SP_MAX; sp++)
    for (int d = 0; d < DIST_MAX; d++)
      if (bins[sp][d])
        printf("%d %d %lld\n", sp, d, bins[sp][d]);
  return 0;
}
