#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { nl = 8 };
struct Row {
  int16_t askRate[nl];
  int16_t bidRate[nl];
  int16_t askSize[nl];
  int16_t bidSize[nl];
  int16_t askNC[nl];
  int16_t bidNC[nl];
  int16_t y;
};

static void swap(int16_t *a, int16_t *b) {
  int16_t tmp[nl];
  memcpy(tmp, a, sizeof tmp);
  memcpy(a, b, sizeof tmp);
  memcpy(b, tmp, sizeof tmp);
}

static void swap_neg(int16_t *a, int16_t *b) {
  int i;
  for (i = 0; i < nl; i++) {
    int16_t t = a[i];
    a[i] = (int16_t)-b[i];
    b[i] = (int16_t)-t;
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
    r.y = (int16_t)-r.y;
    if (fwrite(&r, sizeof r, 1, stdout) != 1) {
      fprintf(stderr, "flip.c: error: fwrite failed\n");
      return 1;
    }
  }
  return 0;
}
