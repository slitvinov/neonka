#include <assert.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "lob.h"

enum { MAXK = nl + 1, STORE_MAX = 100 };
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

int main(int argc, char **argv) {
  int64_t n, nb, na, i, j, p;
  struct Row *rows;
  int64_t store[STORE_MAX];
  int64_t C[MAXK][MAXK] = {0};

  if (argv[1] && strcmp(argv[1], "-h") == 0) {
    puts("usage: split <input.raw> <output.raw>");
    return 0;
  }
  assert(argv[1]);
  assert(argv[2]);
  assert(argv[3] == NULL);

  if (read_raw(argv[1], &rows, &n) != 0)
    return 1;
  p = 0;
  store[p++] = 0;
  for (i = 1; i < n; i++) {
    na = overlap8(rows[i - 1].askRate, rows[i].askRate);
    nb = overlap8(rows[i - 1].bidRate, rows[i].bidRate);
    assert(na < MAXK && nb < MAXK);
    if (na + nb <= 4) {
      assert(p < STORE_MAX - 1);
      store[p++] = i;
    }
    C[na][nb]++;
  }
  store[p++] = n;

  for (i = 0; i < MAXK; i++) {
    for (j = 0; j < MAXK; j++)
      printf("% 9lld ", C[i][j]);
    printf("\n");
  }
  FILE *f = fopen(argv[2], "wb");
  if (f == NULL) {
    fprintf(stderr, "sessions.c: error: fail to open '%s'\n", argv[2]);
    free_raw(rows, n);
    return 1;
  }
  if (fwrite(store, sizeof(int64_t), (size_t)p, f) != (size_t)p) {
    fprintf(stderr, "sessions.c: error: fwrite failed for '%s'\n", argv[2]);
    fclose(f);
    free_raw(rows, n);
    return 1;
  }
  fclose(f);
  free_raw(rows, n);
  return 0;
}
