#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { nl = 8, MAXK = nl + 1 };
struct Row {
  int32_t askRate[nl];
  int32_t bidRate[nl];
  int32_t askSize[nl];
  int32_t bidSize[nl];
  int32_t askNC[nl];
  int32_t bidNC[nl];
  int32_t y;
};

static int overlap8(int32_t *a, int32_t *b) {
  int n = 0, j, k;
  for (j = 0; j < nl; j++) {
    for (k = 0; k < nl; k++)
      if (a[j] == b[k]) {
        n++;
        break;
      }
  }
  return n;
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  int64_t C[MAXK][MAXK] = {0};
  struct Row cur, prev;
  int have_prev = 0;
  int j, k;
  while (fread(&cur, sizeof cur, 1, stdin) == 1) {
    if (have_prev) {
      int na = overlap8(prev.askRate, cur.askRate);
      int nb = overlap8(prev.bidRate, cur.bidRate);
      assert(na < MAXK && nb < MAXK);
      C[na][nb]++;
    }
    if (fwrite(&cur, sizeof cur, 1, stdout) != 1) {
      fprintf(stderr, "overlap.c: error: fwrite failed\n");
      return 1;
    }
    prev = cur;
    have_prev = 1;
  }
  for (j = 0; j < MAXK; j++) {
    for (k = 0; k < MAXK; k++)
      fprintf(stderr, "% 9lld ", (long long)C[j][k]);
    fprintf(stderr, "\n");
  }
  return 0;
}
