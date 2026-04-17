#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { nl = 8, RATE_BASE = 31508 };
struct Row {
  int16_t askRate[nl];
  int16_t bidRate[nl];
  int16_t askSize[nl];
  int16_t bidSize[nl];
  int16_t askNC[nl];
  int16_t bidNC[nl];
  int16_t y;
};

static void emit_rate(int16_t val, int is_nan) {
  int num, q;
  if (is_nan) {
    fputs("NaN", stdout);
    return;
  }
  num = (int)val + RATE_BASE;
  assert(num % 2 == 0);
  q = num / 2;
  if (q % 2 == 0)
    printf("%d", q / 2);
  else
    printf("%d.5", q / 2);
}

static void emit_intnan(int16_t val, int is_nan) {
  if (is_nan)
    fputs("NaN", stdout);
  else
    printf("%d", (int)val);
}

static void emit_y(int16_t val) {
  double d = (double)val / 4.0;
  printf("%g", d);
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  const char *groups[6] = {"askRate", "bidRate", "askSize",
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
      ask_nan[k] = (r.askRate[k] == 0);
      bid_nan[k] = (r.bidRate[k] == 0);
    }
    for (k = 0; k < nl; k++) {
      if (k > 0)
        putchar(',');
      emit_rate(r.askRate[k], ask_nan[k]);
    }
    for (k = 0; k < nl; k++) {
      putchar(',');
      emit_rate(r.bidRate[k], bid_nan[k]);
    }
    for (k = 0; k < nl; k++) {
      putchar(',');
      emit_intnan(r.askSize[k], ask_nan[k]);
    }
    for (k = 0; k < nl; k++) {
      putchar(',');
      emit_intnan(r.bidSize[k], bid_nan[k]);
    }
    for (k = 0; k < nl; k++) {
      putchar(',');
      emit_intnan(r.askNC[k], 0);
    }
    for (k = 0; k < nl; k++) {
      putchar(',');
      emit_intnan(r.bidNC[k], 0);
    }
    putchar(',');
    emit_y(r.y);
    putchar('\n');
  }
  return 0;
}
