#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { nl = 8 };
struct Row {
  int32_t aR[nl];
  int32_t bR[nl];
  int32_t aS[nl];
  int32_t bS[nl];
  int32_t aN[nl];
  int32_t bN[nl];
  int32_t y;
};

static void emit_rate(int32_t val, int is_nan) {
  if (is_nan) {
    fputs("NaN", stdout);
    return;
  }
  assert(val % 2 == 0);
  if (val % 4 == 0)
    printf("%d", val / 4);
  else
    printf("%d.5", val / 4);
}

static void emit_intnan(int32_t val, int is_nan) {
  if (is_nan)
    fputs("NaN", stdout);
  else
    printf("%d", (int)val);
}

static void emit_y(int32_t val) {
  double d = (double)val / 4.0;
  printf("%g", d);
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  char *groups[6] = {"askRate", "bidRate", "askSize",
                     "bidSize", "askNc",   "bidNc"};
  int first = 1, g, k;
  struct Row r;
  int ask_nan[nl], bid_nan[nl];

  for (g = 0; g < 6; g++)
    for (k = 0; k < nl; k++) {
      printf("%s%s_%d", first ? "" : ",", groups[g], k);
      first = 0;
    }
  printf(",y\n");

  while (fread(&r, sizeof r, 1, stdin) == 1) {
    for (k = 0; k < nl; k++) {
      ask_nan[k] = (r.aR[k] == 0);
      bid_nan[k] = (r.bR[k] == 0);
    }
    for (k = 0; k < nl; k++) {
      if (k > 0)
        putchar(',');
      emit_rate(r.aR[k], ask_nan[k]);
    }
    for (k = 0; k < nl; k++) {
      putchar(',');
      emit_rate(r.bR[k], bid_nan[k]);
    }
    for (k = 0; k < nl; k++) {
      putchar(',');
      emit_intnan(r.aS[k], ask_nan[k]);
    }
    for (k = 0; k < nl; k++) {
      putchar(',');
      emit_intnan(r.bS[k], bid_nan[k]);
    }
    for (k = 0; k < nl; k++) {
      putchar(',');
      emit_intnan(r.aN[k], 0);
    }
    for (k = 0; k < nl; k++) {
      putchar(',');
      emit_intnan(r.bN[k], 0);
    }
    putchar(',');
    emit_y(r.y);
    putchar('\n');
  }
  return 0;
}
