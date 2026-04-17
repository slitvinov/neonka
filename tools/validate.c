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
  (void)argc;
  (void)argv;
  struct Row r;
  long long nr = 0;
  long long nzm = 0, nncs = 0, nsrt = 0, ncross = 0;
  char *gnames[6] = {"askRate", "bidRate", "askSize",
                           "bidSize", "askNC",   "bidNC"};
  while (fread(&r, sizeof r, 1, stdin) == 1) {
    int32_t *groups[6] = {r.askRate, r.bidRate, r.askSize,
                          r.bidSize, r.askNC,   r.bidNC};
    int g, j;
    for (g = 0; g < 6; g++) {
      int saw_zero = 0;
      for (j = 0; j < nl; j++) {
        if (groups[g][j] == 0)
          saw_zero = 1;
        else if (saw_zero) {
          if (nzm < 10)
            fprintf(stderr,
                    "zero-mono row %lld: %s_%d=%d after a zero in group\n",
                    nr + 1, gnames[g], j, groups[g][j]);
          nzm++;
        }
      }
    }
    for (j = 0; j < nl; j++) {
      if (r.askNC[j] > r.askSize[j]) {
        if (nncs < 10)
          fprintf(stderr, "NC>Size row %lld: askNC_%d=%d > askSize_%d=%d\n",
                  nr + 1, j, r.askNC[j], j, r.askSize[j]);
        nncs++;
      }
      if (r.bidNC[j] > r.bidSize[j]) {
        if (nncs < 10)
          fprintf(stderr, "NC>Size row %lld: bidNC_%d=%d > bidSize_%d=%d\n",
                  nr + 1, j, r.bidNC[j], j, r.bidSize[j]);
        nncs++;
      }
    }
    for (j = 1; j < nl; j++) {
      if (r.askRate[j] != 0 && r.askRate[j - 1] != 0 &&
          r.askRate[j] <= r.askRate[j - 1]) {
        if (nsrt < 10)
          fprintf(stderr,
                  "sort row %lld: askRate_%d=%d <= askRate_%d=%d\n",
                  nr + 1, j, r.askRate[j], j - 1, r.askRate[j - 1]);
        nsrt++;
      }
      if (r.bidRate[j] != 0 && r.bidRate[j - 1] != 0 &&
          r.bidRate[j] >= r.bidRate[j - 1]) {
        if (nsrt < 10)
          fprintf(stderr,
                  "sort row %lld: bidRate_%d=%d >= bidRate_%d=%d\n",
                  nr + 1, j, r.bidRate[j], j - 1, r.bidRate[j - 1]);
        nsrt++;
      }
    }
    if (r.askRate[0] != 0 && r.bidRate[0] != 0 &&
        r.askRate[0] < r.bidRate[0]) {
      if (ncross < 10)
        fprintf(stderr, "crossed row %lld: askRate_0=%d < bidRate_0=%d\n",
                nr + 1, r.askRate[0], r.bidRate[0]);
      ncross++;
    }
    if (fwrite(&r, sizeof r, 1, stdout) != 1) {
      fprintf(stderr, "validate.c: error: fwrite failed\n");
      return 1;
    }
    nr++;
  }
  fprintf(stderr,
          "validate: records=%lld zero-mono=%lld NC>Size=%lld "
          "sort=%lld crossed=%lld\n",
          nr, nzm, nncs, nsrt, ncross);
  return 0;
}
