#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { nl = 8, MAXK = nl + 1 };
enum { MIN_SESSION_LEN = 150000 };
struct Row {
  int32_t aR[nl], bR[nl], aS[nl], bS[nl], aN[nl], bN[nl], y;
};

static int overlap8(int32_t *a, int32_t *b) {
  int n = 0;
  for (int j = 0; j < nl; j++) {
    for (int k = 0; k < nl; k++)
      if (a[j] == b[k] && a[j] != 0 && a[k] != 0) {
        n++;
        break;
      }
  }
  return n;
}

static void emit_i64(int64_t v) {
  if (fwrite(&v, sizeof v, 1, stdout) != 1) {
    fprintf(stderr, "split.c: error: fwrite failed\n");
    exit(1);
  }
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  struct Row cur, prev;
  int64_t i = 0;
  int have_prev = 0;
  int64_t last_boundary = 0;
  int64_t C[MAXK][MAXK] = {0};
  emit_i64(0);
  while (fread(&cur, sizeof cur, 1, stdin) == 1) {
    if (have_prev) {
      int na = overlap8(prev.aR, cur.aR);
      int nb = overlap8(prev.bR, cur.bR);
      assert(na < MAXK && nb < MAXK);
      C[na][nb]++;
      if (na + nb <= 4 && i - last_boundary >= MIN_SESSION_LEN) {
        emit_i64(i);
        last_boundary = i;
      }
    }
    prev = cur;
    have_prev = 1;
    i++;
  }
  emit_i64(i);
  for (int r = 0; r < MAXK; r++) {
    for (int c = 0; c < MAXK; c++)
      fprintf(stderr, "% 9lld ", (long long)C[r][c]);
    fprintf(stderr, "\n");
  }
  return 0;
}
