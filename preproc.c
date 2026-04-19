/* preproc — extract 6-D pooled event stream for hawkes.c.
 *
 * Reads train.events for one session, pools ask/bid into single types,
 * splits tm by pre-event N[0]. Writes binary (int32 t, int32 type) records.
 *
 * Pooled 6-D taxonomy:
 *   0  tp (ask+bid pooled)
 *   1  tm_queue  (tm fired when pre-event N[0] > 1; no cascade)
 *   2  tm_cascade(tm fired when pre-event N[0] = 1; cascade + refill)
 *   3  dp (ask+bid pooled)
 *   4  dm (ask+bid pooled)
 *   5  hp (ask+bid pooled; observation artifact)
 *
 * Usage:  ./preproc -D train.events -S sessions.events.raw -s <session_id>
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { REC_COLS = 54, EV_IDLE = 8 };

int main(int argc, char **argv) {
  const char *events_path = NULL, *idx_path = NULL;
  long session_id = -1;
  for (int i = 1; i < argc; i++) {
    if      (!strcmp(argv[i], "-D") && i + 1 < argc) events_path = argv[++i];
    else if (!strcmp(argv[i], "-S") && i + 1 < argc) idx_path    = argv[++i];
    else if (!strcmp(argv[i], "-s") && i + 1 < argc) session_id  = atol(argv[++i]);
    else { fprintf(stderr, "preproc: unknown arg '%s'\n", argv[i]); return 1; }
  }
  if (!events_path || !idx_path || session_id < 0) {
    fprintf(stderr, "usage: preproc -D events -S idx -s session_id\n");
    return 1;
  }

  FILE *f = fopen(events_path, "rb");
  if (!f) { fprintf(stderr, "preproc: cannot open %s\n", events_path); return 1; }
  FILE *sf = fopen(idx_path, "rb");
  if (!sf) { fprintf(stderr, "preproc: cannot open %s\n", idx_path); return 1; }

  int64_t off[2];
  if (fseeko(sf, (off_t)session_id * (off_t)sizeof(int64_t), SEEK_SET) != 0 ||
      fread(off, sizeof(int64_t), 2, sf) != 2) {
    fprintf(stderr, "preproc: idx seek/read failed\n"); return 1;
  }
  fclose(sf);

  if (fseeko(f, off[0], SEEK_SET) != 0) {
    fprintf(stderr, "preproc: seek to session %ld failed\n", session_id); return 1;
  }
  off_t bytes_remaining = (off_t)(off[1] - off[0]);

  const off_t recsz = REC_COLS * (off_t)sizeof(int32_t);
  int32_t w[REC_COLS];
  long n_ev = 0, n_tm_q = 0, n_tm_c = 0;
  while (bytes_remaining >= recsz) {
    if (fread(w, recsz, 1, f) != 1) break;
    bytes_remaining -= recsz;
    int32_t t     = w[0];            /* event type (0..7 event, 8 IDLE) */
    int32_t orig  = w[1];            /* row index = time */
    int32_t aN0   = w[5 + 32];       /* col 37: aN[0] */
    int32_t bN0   = w[5 + 40];       /* col 45: bN[0] */
    if (t == EV_IDLE) continue;      /* skip IDLE markers */

    int32_t pooled;
    if      (t == 0 || t == 1) pooled = 0;                              /* tp */
    else if (t == 2)           pooled = (aN0 > 1 ? 1 : 2);              /* tm_a */
    else if (t == 3)           pooled = (bN0 > 1 ? 1 : 2);              /* tm_b */
    else if (t == 4 || t == 5) pooled = 3;                              /* dp */
    else if (t == 6 || t == 7) pooled = 4;                              /* dm */
    else if (t >= 9)           pooled = 5;                              /* hp */
    else                       continue;

    int32_t rec[2] = { orig, pooled };
    if (fwrite(rec, sizeof rec, 1, stdout) != 1) {
      fprintf(stderr, "preproc: write failed\n"); return 1;
    }
    n_ev++;
    if (pooled == 1) n_tm_q++;
    if (pooled == 2) n_tm_c++;
  }
  fclose(f);
  long n_tm = n_tm_q + n_tm_c;
  fprintf(stderr, "preproc ses%ld: %ld events  tm_q=%ld  tm_c=%ld  "
          "cascade frac = %.4f\n",
          session_id, n_ev, n_tm_q, n_tm_c,
          (n_tm > 0 ? (double)n_tm_c / (double)n_tm : 0.0));
  return 0;
}
