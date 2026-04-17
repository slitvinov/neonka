#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
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

int main(int argc, char **argv) {
  long n = 0;
  int i;
  for (i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
      n = atol(argv[++i]);
    else {
      fprintf(stderr, "usage: stride -n N < <input.raw>\n");
      return 1;
    }
  }
  if (n < 1) {
    fprintf(stderr, "stride.c: error: need -n N with N >= 1\n");
    return 1;
  }
  struct Row row;
  long k;
  for (;;) {
    if (fread(&row, sizeof row, 1, stdin) != 1)
      return 0;
    if (fwrite(&row, sizeof row, 1, stdout) != 1) {
      fprintf(stderr, "stride.c: error: fwrite failed\n");
      return 1;
    }
    for (k = 1; k < n; k++) {
      if (fread(&row, sizeof row, 1, stdin) != 1)
        return 0;
    }
  }
}
