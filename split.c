#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { nl = 8, MAXK = nl + 1 };
struct Row {
  int32_t aR[nl];
  int32_t bR[nl];
  int32_t aS[nl];
  int32_t bS[nl];
  int32_t aN[nl];
  int32_t bN[nl];
  int32_t y;
};

static int overlap8(int32_t *a, int32_t *b) {
  int n = 0;
  for (int j = 0; j < nl; j++) {
    for (int k = 0; k < nl; k++)
      if (a[j] == b[k]) {
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
  emit_i64(0);
  while (fread(&cur, sizeof cur, 1, stdin) == 1) {
    if (have_prev) {
      int na = overlap8(prev.aR, cur.aR);
      int nb = overlap8(prev.bR, cur.bR);
      assert(na < MAXK && nb < MAXK);
      if (na + nb <= 4)
        emit_i64(i);
    }
    prev = cur;
    have_prev = 1;
    i++;
  }
  emit_i64(i);
  return 0;
}
