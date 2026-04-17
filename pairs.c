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
  struct Row cur, prev;
  int have = 0;
  while (fread(&cur, sizeof cur, 1, stdin) == 1) {
    if (have) {
      if (fwrite(&prev, sizeof prev, 1, stdout) != 1 ||
          fwrite(&cur, sizeof cur, 1, stdout) != 1) {
        fprintf(stderr, "pairs.c: error: fwrite failed\n");
        return 1;
      }
    }
    prev = cur;
    have = 1;
  }
  return 0;
}
