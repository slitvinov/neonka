/* events.c — extract elementary event log from session LOB snapshots.
 *
 * Input  (stdin): raw Row stream, 49 int32/row (output of session / train.raw).
 * Output (stdout): packed { int32 t; int32 type; } per elementary event.
 *
 * Event types:
 *   0 tp_a  1 tp_b  2 tm_a  3 tm_b
 *   4 dp_a  5 dp_b  6 dm_a  7 dm_b
 *   8 hp_a  9 hp_b   (hidden-surface insertion: top-level cascade refilled level 7
 *                     with a new price — observed footprint of a hidden-book event)
 *
 * t is the row index (event clock). Multi-event frames emit multiple records
 * at the same t in source-order (ask side first, then bid side).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { nl = 8 };
struct Row {
  int32_t aR[nl], bR[nl], aS[nl], bS[nl], aN[nl], bN[nl], y;
};

static void emit(int32_t t, int32_t type) {
  int32_t rec[2] = { t, type };
  if (fwrite(rec, sizeof rec, 1, stdout) != 1) {
    fprintf(stderr, "events.c: error: fwrite failed\n");
    exit(1);
  }
}

/* Walk one side, emit elementary events at time t.
 * side: 0 = ask (ascending R), 1 = bid (descending R).
  }
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  struct Row prev, cur;
  int32_t t = 0;
  int have = 0;
  while (fread(&cur, sizeof cur, 1, stdin) == 1) {
    if (have) {
      walk_side(t, prev.aR, prev.aN, cur.aR, cur.aN, 0);
      walk_side(t, prev.bR, prev.bN, cur.bR, cur.bN, 1);
    }
    prev = cur;
    have = 1;
    t++;
  }
  return 0;
}
