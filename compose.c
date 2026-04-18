/* compose.c — inverse of decompose: rebuild train.raw rows from train.events.
 *
 * Reads 54-int32 event records. For each IDLE record (type==8), writes the
 * 49-int32 book to stdout. IDLE markers are in orig_idx order, so the
 * output matches train.raw byte-for-byte.
 *
 * Default: reads from stdin (or -D events), emits all rows.
 * With -D events -S idx -s N: seeks to session N's byte-offset range.
 *
 * Usage:
 *   ./compose < data/train.events > data/train.raw
 *   ./compose -D data/train.events -S data/sessions.events.raw -s 0 > session0.raw
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { REC = 54, BOOK_OFF = 5, BOOK_LEN = 49, E_IDLE = 8 };

int main(int argc, char **argv) {
  char *dpath = NULL, *spath = NULL;
  long sid = -1;
  int i;
  for (i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-D") && i+1 < argc) dpath = argv[++i];
    else if (!strcmp(argv[i], "-S") && i+1 < argc) spath = argv[++i];
    else if (!strcmp(argv[i], "-s") && i+1 < argc) sid = atol(argv[++i]);
    else { fprintf(stderr, "usage: compose [-D events [-S idx -s N]] > rows.raw\n"); return 1; }
  }

  FILE *in = dpath ? fopen(dpath, "rb") : stdin;
  if (!in) { fprintf(stderr, "compose: cannot open %s\n", dpath); return 1; }

  long long stop_bytes = -1;  /* -1 = unlimited */
  if (sid >= 0) {
    if (!spath) { fprintf(stderr, "compose: -s requires -S <idx_file>\n"); return 1; }
    FILE *sf = fopen(spath, "rb");
    if (!sf) { fprintf(stderr, "compose: cannot open %s\n", spath); return 1; }
    int64_t off[2];
    if (fseeko(sf, sid * sizeof(int64_t), SEEK_SET) != 0 ||
        fread(off, sizeof(int64_t), 2, sf) != 2) {
      fprintf(stderr, "compose: seek/read idx failed\n"); return 1;
    }
    fclose(sf);
    if (fseeko(in, off[0], SEEK_SET) != 0) {
      fprintf(stderr, "compose: seek to session offset failed\n"); return 1;
    }
    stop_bytes = off[1] - off[0];
  }

  int32_t rec[REC];
  long long n_out = 0, bytes_read = 0;
  while (fread(rec, sizeof rec, 1, in) == 1) {
    if (stop_bytes >= 0) {
      bytes_read += sizeof rec;
      if (bytes_read > stop_bytes) break;
    }
    if (rec[0] == E_IDLE) {
      if (fwrite(&rec[BOOK_OFF], sizeof(int32_t), BOOK_LEN, stdout) != BOOK_LEN) {
        fprintf(stderr, "compose: fwrite failed\n"); return 1;
      }
      n_out++;
    }
  }
  if (dpath) fclose(in);
  fprintf(stderr, "compose: %lld rows written\n", n_out);
  return 0;
}
