#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { nl = 8 };
struct Row {
  int32_t askRate[nl];
  int32_t bidRate[nl];
  int32_t askSize[nl];
  int32_t bidSize[nl];
  int32_t askNC[nl];
  int32_t bidNC[nl];
  int32_t y;
};

static void swap(int32_t *a, int32_t *b) {
  int32_t tmp[nl];
  memcpy(tmp, a, sizeof tmp);
  memcpy(a, b, sizeof tmp);
  memcpy(b, tmp, sizeof tmp);
}

static void swap_neg(int32_t *a, int32_t *b) {
  int i;
  for (i = 0; i < nl; i++) {
    int32_t t = a[i];
    a[i] = (int32_t)-b[i];
    b[i] = (int32_t)-t;
  }
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  struct Row r;
  while (fread(&r, sizeof r, 1, stdin) == 1) {
    swap_neg(r.askRate, r.bidRate);
    swap(r.askSize, r.bidSize);
    swap(r.askNC, r.bidNC);
    r.y = (int32_t)-r.y;
    if (fwrite(&r, sizeof r, 1, stdout) != 1) {
      fprintf(stderr, "flip.c: error: fwrite failed\n");
      return 1;
    }
  }
  return 0;
}
