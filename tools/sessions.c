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

enum { nl = 8, MAXK = nl + 1, STORE_MAX = 100 };
struct Row {
  int16_t askRate[nl];
  int16_t bidRate[nl];
  int16_t askSize[nl];
  int16_t bidSize[nl];
  int16_t askNC[nl];
  int16_t bidNC[nl];
  int16_t y;
};

static int read_raw(const char *path, struct Row **rows, int64_t *n) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) {
    fprintf(stderr, "sessions.c: error: failed to open '%s'\n", path);
    return -1;
  }
  struct stat st;
  if (fstat(fd, &st) != 0) {
    fprintf(stderr, "sessions.c: error: fstat failed for '%s'\n", path);
    close(fd);
    return -1;
  }
  int64_t bytes = (int64_t)st.st_size;
  if (bytes % (int64_t)sizeof(struct Row) != 0) {
    fprintf(stderr, "sessions.c: error: bad file size '%s'\n", path);
    close(fd);
    return -1;
  }
  void *map = mmap(NULL, (size_t)bytes, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (map == MAP_FAILED) {
    fprintf(stderr, "sessions.c: error: mmap failed for '%s'\n", path);
    return -1;
  }
  *rows = map;
  *n = bytes / (int64_t)sizeof(struct Row);
  return 0;
}

static void free_raw(struct Row *rows, int64_t n) {
  munmap(rows, (size_t)n * sizeof(struct Row));
}

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
