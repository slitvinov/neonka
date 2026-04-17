#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { nl = 8 };
struct Row {
  int32_t aR[nl], bR[nl], aS[nl], bS[nl], aN[nl], bN[nl], y;
};

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  int64_t prev_tick = -1;
  struct Row first, last, r;
  int64_t tick;
  int have = 0;
  while (fread(&tick, sizeof tick, 1, stdin) == 1 &&
         fread(&r, sizeof r, 1, stdin) == 1) {
    if (tick != prev_tick) {
      if (have) {
        fwrite(&first, sizeof first, 1, stdout);
        fwrite(&last, sizeof last, 1, stdout);
      }
      first = r;
      last = r;
      prev_tick = tick;
      have = 1;
    } else {
      last = r;
    }
  }
  if (have) {
    fwrite(&first, sizeof first, 1, stdout);
    fwrite(&last, sizeof last, 1, stdout);
  }
  return 0;
}
