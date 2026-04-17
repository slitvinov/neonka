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
  struct Row r;
  long long nr = 0;
  long long nzm = 0, nncs = 0, nsrt = 0, ncross = 0;
  char *gnames[6] = {"aR", "bR", "aS",
                           "bS", "aN",   "bN"};
  while (fread(&r, sizeof r, 1, stdin) == 1) {
    int32_t *groups[6] = {r.aR, r.bR, r.aS,
                          r.bS, r.aN,   r.bN};
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
      if (r.aN[j] > r.aS[j]) {
        if (nncs < 10)
          fprintf(stderr, "NC>Size row %lld: aN_%d=%d > aS_%d=%d\n",
                  nr + 1, j, r.aN[j], j, r.aS[j]);
        nncs++;
      }
      if (r.bN[j] > r.bS[j]) {
        if (nncs < 10)
          fprintf(stderr, "NC>Size row %lld: bN_%d=%d > bS_%d=%d\n",
                  nr + 1, j, r.bN[j], j, r.bS[j]);
        nncs++;
      }
    }
    for (j = 1; j < nl; j++) {
      if (r.aR[j] != 0 && r.aR[j - 1] != 0 &&
          r.aR[j] <= r.aR[j - 1]) {
        if (nsrt < 10)
          fprintf(stderr,
                  "sort row %lld: aR_%d=%d <= aR_%d=%d\n",
                  nr + 1, j, r.aR[j], j - 1, r.aR[j - 1]);
        nsrt++;
      }
      if (r.bR[j] != 0 && r.bR[j - 1] != 0 &&
          r.bR[j] >= r.bR[j - 1]) {
        if (nsrt < 10)
          fprintf(stderr,
                  "sort row %lld: bR_%d=%d >= bR_%d=%d\n",
                  nr + 1, j, r.bR[j], j - 1, r.bR[j - 1]);
        nsrt++;
      }
    }
    if (r.aR[0] != 0 && r.bR[0] != 0 &&
        r.aR[0] < r.bR[0]) {
      if (ncross < 10)
        fprintf(stderr, "crossed row %lld: aR_0=%d < bR_0=%d\n",
                nr + 1, r.aR[0], r.bR[0]);
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
