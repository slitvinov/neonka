/* halftick.c — double rates and y
 * Usage: halftick < <input.raw> > <output.raw>
 * Multiplies askRate, bidRate, and y by 2.  Leaves sizes and NCs alone
 * (they are counts, not prices).  Useful to align scales: after halftick,
 * (askRate[0] + bidRate[0]) / 2 is always an integer (mid centers exactly
 * because the spread in doubled units is always even).
 */
#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
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

static int16_t dbl(int16_t v) {
  int32_t t = 2 * (int32_t)v;
  assert(t >= INT16_MIN && t <= INT16_MAX);
  return (int16_t)t;
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  struct Row r;
  int i;
  while (fread(&r, sizeof r, 1, stdin) == 1) {
    for (i = 0; i < nl; i++) {
      r.askRate[i] = dbl(r.askRate[i]);
      r.bidRate[i] = dbl(r.bidRate[i]);
    }
    r.y = dbl(r.y);
    if (fwrite(&r, sizeof r, 1, stdout) != 1) {
      fprintf(stderr, "halftick.c: error: fwrite failed\n");
      return 1;
    }
  }
  return 0;
}
