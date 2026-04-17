#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { nl = 8, MAXK = nl + 1 };
struct Row {
  int16_t askRate[nl];
  int16_t bidRate[nl];
  int16_t askSize[nl];
  int16_t bidSize[nl];
  int16_t askNC[nl];
  int16_t bidNC[nl];
  int16_t y;
};

static int overlap8(const int16_t *a, const int16_t *b) {
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
      int na = overlap8(prev.askRate, cur.askRate);
      int nb = overlap8(prev.bidRate, cur.bidRate);
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
