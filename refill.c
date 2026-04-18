/* refill.c — extract cascade refill distance distribution from row pairs.
 *
 * At each visible-book cascade (top level emptied → all levels shift up →
 * a hidden-book level surfaces at R[NL-1]), record cur.R[NL-1] - cur.R[NL-2].
 * This is the empirical marginal of hidden-queue-surface offsets — a
 * data-driven stand-in for latent hidden-book dynamics.
 *
 * Input (stdin):  standard 49-int32 row stream (session output).
 * Output (stdout):
 *   ask: encoded-tick distance, count
 *   bid: same (negated encoded ticks → positive tick units)
 *
 * Usage: session ... | refill > refill_hist.txt
 *   (or: ./compose -D data/train.events -S sessions.events.raw -s S | refill)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { nl = 8, MAX_DIST = 256 };
struct Row {
  int32_t aR[nl], bR[nl], aS[nl], bS[nl], aN[nl], bN[nl], y;
};

static long long hist_a[MAX_DIST];
static long long hist_b[MAX_DIST];

/* Cascade detected on one side if prev's top-level entry is gone in cur
 * (i.e., the walk encounters d>0 at i=0). To keep logic clean, we detect
 * by comparing the top prices: if cur.R[0] != prev.R[0] but prev.R[0] is
 * present in cur as R[1] or deeper, that's a shift-up — cascade happened. */
/* Cascade on ask = prev's top level gone, cur's top is prev's 2nd level.
 * This is the signature of "apply_tm on ask emptied aN[0] and shifted up". */
static int detect_cascade_ask(int32_t *pR, int32_t *pN, int32_t *cR, int32_t *cN) {
  (void)cN;
  if (pN[0] == 0 || pN[1] == 0) return 0;
  return cR[0] == pR[1];
}
static int detect_cascade_bid(int32_t *pR, int32_t *pN, int32_t *cR, int32_t *cN) {
  (void)cN;
  if (pN[0] == 0 || pN[1] == 0) return 0;
  return cR[0] == pR[1];
}

int main(int argc, char **argv) {
  (void)argc; (void)argv;
  struct Row prev, cur;
  int have = 0;
  long long n_pairs = 0, n_cascade_a = 0, n_cascade_b = 0;
  while (fread(&cur, sizeof cur, 1, stdin) == 1) {
    if (have) {
      n_pairs++;
      if (cur.aN[nl-1] != 0 && prev.aN[nl-1] != 0 &&
          detect_cascade_ask(prev.aR, prev.aN, cur.aR, cur.aN)) {
        int32_t dist = cur.aR[nl-1] - cur.aR[nl-2];
        if (dist >= 0 && dist < MAX_DIST) hist_a[dist]++;
        n_cascade_a++;
      }
      if (cur.bN[nl-1] != 0 && prev.bN[nl-1] != 0 &&
          detect_cascade_bid(prev.bR, prev.bN, cur.bR, cur.bN)) {
        int32_t dist = cur.bR[nl-2] - cur.bR[nl-1];
        if (dist >= 0 && dist < MAX_DIST) hist_b[dist]++;
        n_cascade_b++;
      }
    }
    prev = cur;
    have = 1;
  }
  /* Output: two-column "dist count", ask then separator then bid */
  int i;
  fprintf(stdout, "# side=a %lld cascades over %lld pairs\n", n_cascade_a, n_pairs);
  for (i = 0; i < MAX_DIST; i++)
    if (hist_a[i] > 0) fprintf(stdout, "a %d %lld\n", i, hist_a[i]);
  fprintf(stdout, "# side=b %lld cascades\n", n_cascade_b);
  for (i = 0; i < MAX_DIST; i++)
    if (hist_b[i] > 0) fprintf(stdout, "b %d %lld\n", i, hist_b[i]);
  fprintf(stderr, "refill: %lld pairs, %lld ask cascades, %lld bid cascades\n",
          n_pairs, n_cascade_a, n_cascade_b);
  return 0;
}
