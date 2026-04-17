/* report.c — pass-through raw filter; prints per-column stats at EOF
 * Usage: report < <input.raw> > <output.raw>
 * Passes int16 rows unchanged; at EOF writes to stderr:
 *   col  min  max  zeros  levels
 * where `levels` is the number of distinct int16 values observed.
 */
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { nf = 49, nl = 8 };
struct Row {
  int16_t askRate[nl];
  int16_t bidRate[nl];
  int16_t askSize[nl];
  int16_t bidSize[nl];
  int16_t askNC[nl];
  int16_t bidNC[nl];
  int16_t y;
};

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  enum { NBIN = 65536, OFFSET = 32768 };
  long long(*hist)[NBIN] = calloc(nf, sizeof *hist);
  if (hist == NULL) {
    fprintf(stderr, "report.c: error: calloc failed\n");
    return 1;
  }
  long long minv[nf], maxv[nf], zeros[nf];
  int i;
  for (i = 0; i < nf; i++) {
    minv[i] = LLONG_MAX;
    maxv[i] = LLONG_MIN;
    zeros[i] = 0;
  }
  static char names[nf][16];
  const char *gnames[6] = {"askRate", "bidRate", "askSize",
                           "bidSize", "askNC",   "bidNC"};
  int g, j;
  for (g = 0; g < 6; g++)
    for (j = 0; j < nl; j++)
      snprintf(names[g * nl + j], sizeof names[0], "%s_%d", gnames[g], j);
  snprintf(names[48], sizeof names[0], "y");

  struct Row r;
  long long nr = 0;
  while (fread(&r, sizeof r, 1, stdin) == 1) {
    const int16_t *rp = (const int16_t *)&r;
    for (i = 0; i < nf; i++) {
      long lv = rp[i];
      if (lv == 0)
        zeros[i]++;
      if (lv < minv[i])
        minv[i] = lv;
      if (lv > maxv[i])
        maxv[i] = lv;
      hist[i][lv + OFFSET]++;
    }
    if (fwrite(&r, sizeof r, 1, stdout) != 1) {
      fprintf(stderr, "report.c: error: fwrite failed\n");
      free(hist);
      return 1;
    }
    nr++;
  }
  fprintf(stderr, "records: %lld\n", nr);
  fprintf(stderr, "col/min/max/zeros/levels\n");
  for (i = 0; i < nf; i++) {
    int levels = 0;
    for (j = 0; j < NBIN; j++)
      if (hist[i][j])
        levels++;
    if (i > 0 && i % nl == 0)
      fprintf(stderr, "\n");
    fprintf(stderr, "%-10s %10lld %10lld %8lld %8d\n", names[i], minv[i],
            maxv[i], zeros[i], levels);
  }
  free(hist);
  return 0;
}
