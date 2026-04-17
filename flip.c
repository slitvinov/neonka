#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { nl = 8 };
struct Row {
  int32_t aR[nl], bR[nl], aS[nl], bS[nl], aN[nl], bN[nl], y;
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
    swap_neg(r.aR, r.bR);
    swap(r.aS, r.bS);
    swap(r.aN, r.bN);
    r.y = (int32_t)-r.y;
    if (fwrite(&r, sizeof r, 1, stdout) != 1) {
      fprintf(stderr, "flip.c: error: fwrite failed\n");
      return 1;
    }
  }
  return 0;
}
