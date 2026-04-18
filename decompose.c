/* decompose.c — per-elementary-event decomposition of train.raw.
 *
 * Self-contained output stream; downstream tools never need train.raw.
 *
 * Record layout (54 int32 = 216 bytes):
 *   [0] type        0..7 = tp_a tp_b tm_a tm_b dp_a dp_b dm_a dm_b ; 8 = IDLE
 *   [1] orig_idx    row in train.raw
 *   [2] level       0..7 affected level (0 for IDLE)
 *   [3] distance    encoded ticks from prev-row best (tp/dp); 0 otherwise
 *   [4] y           label at orig_idx (= train.raw[orig_idx].y); same across
 *                   all records sharing the same orig_idx (for fast lookup
 *                   without dereferencing book[48])
 *   [5..53] book    state BEFORE this event fires; for IDLE markers = observed state
 *
 * Pass 1 (naive): per row transition, emit each elementary event with book
 * = prev-row state (no intermediate-state tracking). Then emit one trailing
 * IDLE with book = cur-row state. This gives clean reconstruction:
 * train.raw[orig_idx] = book of IDLE record at orig_idx.
 *
 * Session first row gets an IDLE record with book = that row. Cross-session
 * transitions are skipped (no events emitted across sessions).
 *
 * Pass 2 (EM refinement) for multi-event rows: TODO. Pass 1 semantics are
 * self-consistent for any downstream calibration (all events in a multi-
 * event row attribute to prev-row state, matching current rates.c behavior).
 *
 * Usage: decompose -D train.raw -S sessions.raw -o out.events
 */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

enum { nl = 8 };
struct Row {
  int32_t aR[nl], bR[nl], aS[nl], bS[nl], aN[nl], bN[nl], y;
};
enum {
  E_TP_A=0, E_TP_B=1, E_TM_A=2, E_TM_B=3,
  E_DP_A=4, E_DP_B=5, E_DM_A=6, E_DM_B=7, E_IDLE=8
};

struct Elem { int type, level, distance; };

static int walk_side(const int32_t *pR, const int32_t *pN,
                     const int32_t *cR, const int32_t *cN,
                     int side, struct Elem *out) {
  int i = 0, j = 0, n = 0, k;
  int tp_t = side ? E_TP_B : E_TP_A;
  int tm_t = side ? E_TM_B : E_TM_A;
  int dp_t = side ? E_DP_B : E_DP_A;
  int dm_t = side ? E_DM_B : E_DM_A;
  while (i < nl && j < nl && pN[i] != 0 && cN[j] != 0) {
    int32_t d = side ? (pR[i] - cR[j]) : (cR[j] - pR[i]);
    if (d < 0) {
      int32_t dist = side ? (cR[j] - pR[i]) : (pR[i] - cR[j]);
      for (k = 0; k < cN[j]; k++) {
        out[n].type = (j == 0) ? tp_t : dp_t;
        out[n].level = j;
        out[n].distance = (j == 0) ? dist : 0;
        n++;
      }
      j++;
    } else if (d == 0) {
      int32_t dn = cN[j] - pN[i];
      if (dn > 0) {
        for (k = 0; k < dn; k++) {
          out[n].type = (j == 0) ? tp_t : dp_t;
          out[n].level = j;
          out[n].distance = 0;
          n++;
        }
      } else if (dn < 0) {
        for (k = 0; k < -dn; k++) {
          out[n].type = (i == 0) ? tm_t : dm_t;
          out[n].level = i;
          out[n].distance = 0;
          n++;
        }
      }
      i++; j++;
    } else {
      for (k = 0; k < pN[i]; k++) {
        out[n].type = (i == 0) ? tm_t : dm_t;
        out[n].level = i;
        out[n].distance = 0;
        n++;
      }
      i++;
    }
  }
  return n;
}

static void emit(FILE *f, int type, int orig_idx, int level, int distance,
                 int32_t y, const struct Row *book) {
  int32_t hdr[5] = { type, orig_idx, level, distance, y };
  if (fwrite(hdr, sizeof hdr, 1, f) != 1 ||
      fwrite(book, sizeof *book, 1, f) != 1) {
    fprintf(stderr, "decompose: fwrite failed\n"); exit(1);
  }
}

int main(int argc, char **argv) {
  char *dpath = NULL, *spath = NULL, *opath = NULL;
  int i;
  for (i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-D") && i+1 < argc) dpath = argv[++i];
    else if (!strcmp(argv[i], "-S") && i+1 < argc) spath = argv[++i];
    else if (!strcmp(argv[i], "-o") && i+1 < argc) opath = argv[++i];
    else { fprintf(stderr, "usage: decompose -D train.raw -S sessions.raw -o out.events\n"); return 1; }
  }
  if (!dpath || !spath || !opath) {
    fprintf(stderr, "decompose: -D, -S, -o required\n"); return 1;
  }
  FILE *sf = fopen(spath, "rb");
  if (!sf) { fprintf(stderr, "cannot open %s\n", spath); return 1; }
  struct stat st; fstat(fileno(sf), &st);
  long nb = st.st_size / sizeof(int64_t);
  int64_t *bounds = malloc(nb * sizeof(int64_t));
  fread(bounds, sizeof(int64_t), nb, sf);
  fclose(sf);
  long nsess = nb - 1;
  fprintf(stderr, "decompose: %ld sessions\n", nsess);

  FILE *in = fopen(dpath, "rb");
  FILE *out = fopen(opath, "wb");
  if (!in || !out) { fprintf(stderr, "open failed\n"); return 1; }

  int64_t *sess_offsets = malloc((nsess + 1) * sizeof(int64_t));
  long long n_events = 0, n_idle = 0, n_multi = 0;
  struct Row prev, cur;
  struct Elem buf_a[128], buf_b[128];
  for (long s = 0; s < nsess; s++) {
    sess_offsets[s] = ftello(out);
    int64_t lo = bounds[s], hi = bounds[s+1];
    fseeko(in, (off_t)lo * (off_t)sizeof prev, SEEK_SET);
    if (fread(&prev, sizeof prev, 1, in) != 1) break;
    /* Session-start snapshot as IDLE at lo; y = prev.y = train.raw[lo].y */
    emit(out, E_IDLE, (int)lo, 0, 0, prev.y, &prev);
    n_events++; n_idle++;
    for (int64_t r = lo + 1; r < hi; r++) {
      if (fread(&cur, sizeof cur, 1, in) != 1) break;
      int na = walk_side(prev.aR, prev.aN, cur.aR, cur.aN, 0, buf_a);
      int nb = walk_side(prev.bR, prev.bN, cur.bR, cur.bN, 1, buf_b);
      int ntot = na + nb;
      if (ntot > 1) n_multi++;
      /* y of the row this transition produces = cur.y = train.raw[r].y */
      int32_t y_r = cur.y;
      int k;
      for (k = 0; k < na; k++) {
        emit(out, buf_a[k].type, (int)r, buf_a[k].level, buf_a[k].distance, y_r, &prev);
        n_events++;
      }
      for (k = 0; k < nb; k++) {
        emit(out, buf_b[k].type, (int)r, buf_b[k].level, buf_b[k].distance, y_r, &prev);
        n_events++;
      }
      emit(out, E_IDLE, (int)r, 0, 0, y_r, &cur);
      n_events++; n_idle++;
      prev = cur;
    }
  }
  sess_offsets[nsess] = ftello(out);
  fclose(in); fclose(out);

  /* Derive index path: data/train.events -> data/sessions.events.raw
   * (mirrors sessions.raw which indexes train.raw). */
  char idxpath[4096];
  const char *slash = strrchr(opath, '/');
  size_t dir_len = slash ? (size_t)(slash - opath + 1) : 0;
  snprintf(idxpath, sizeof idxpath, "%.*ssessions.events.raw",
           (int)dir_len, opath);
  FILE *idx = fopen(idxpath, "wb");
  if (!idx) { fprintf(stderr, "cannot open %s\n", idxpath); return 1; }
  if (fwrite(sess_offsets, sizeof(int64_t), nsess + 1, idx) != (size_t)(nsess + 1)) {
    fprintf(stderr, "decompose: idx write failed\n"); return 1;
  }
  fclose(idx);

  fprintf(stderr, "decompose: %lld records, %lld IDLE markers, %lld multi-event rows\n",
          n_events, n_idle, n_multi);
  fprintf(stderr, "decompose: session index %s (%ld + 1 int64 offsets)\n", idxpath, nsess);
  free(bounds);
  free(sess_offsets);
  return 0;
}
