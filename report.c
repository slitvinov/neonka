#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { nf = 49, nl = 8, HCAP = 16384 };

struct Row {
  int32_t aR[nl], bR[nl], aS[nl], bS[nl], aN[nl], bN[nl], y;
};

struct HSet {
  int32_t keys[HCAP];
  char occ[HCAP];
};

static void hset_add(struct HSet *s, int32_t v, int64_t *count) {
  uint32_t i = ((uint32_t)v * 2654435761u) & (HCAP - 1);
  while (s->occ[i]) {
    if (s->keys[i] == v)
      return;
    i = (i + 1) & (HCAP - 1);
  }
  s->keys[i] = v;
  s->occ[i] = 1;
  (*count)++;
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  int64_t minv[nf], maxv[nf], zeros[nf], levels[nf];
  struct HSet *sets = calloc(nf, sizeof *sets);
  if (sets == NULL) {
    fprintf(stderr, "report.c: error: calloc failed\n");
    return 1;
  }
  int i;
  for (i = 0; i < nf; i++) {
    minv[i] = INT64_MAX;
    maxv[i] = INT64_MIN;
    zeros[i] = 0;
    levels[i] = 0;
  }
  static char names[nf][16];
  char *gnames[6] = {"aR", "bR", "aS",
                     "bS", "aN",   "bN"};
  int g, j;
  for (g = 0; g < 6; g++)
    for (j = 0; j < nl; j++)
      snprintf(names[g * nl + j], sizeof names[0], "%s_%d", gnames[g], j);
  snprintf(names[48], sizeof names[0], "y");

  struct Row r;
  int64_t nr = 0;
  while (fread(&r, sizeof r, 1, stdin) == 1) {
    int32_t *rp = (int32_t *)&r;
    for (i = 0; i < nf; i++) {
      int32_t v = rp[i];
      if (v == 0)
        zeros[i]++;
      if (v < minv[i])
        minv[i] = v;
      if (v > maxv[i])
        maxv[i] = v;
      hset_add(&sets[i], v, &levels[i]);
    }
    if (fwrite(&r, sizeof r, 1, stdout) != 1) {
      fprintf(stderr, "report.c: error: fwrite failed\n");
      free(sets);
      return 1;
    }
    nr++;
  }
  fprintf(stderr, "records: %lld\n", (long long)nr);
  fprintf(stderr, "col/min/max/zeros/levels\n");
  for (i = 0; i < nf; i++) {
    if (i > 0 && i % nl == 0)
      fprintf(stderr, "\n");
    fprintf(stderr, "%-10s %10lld %10lld %8lld %8lld\n", names[i],
            (long long)minv[i], (long long)maxv[i], (long long)zeros[i],
            (long long)levels[i]);
  }
  free(sets);
  return 0;
}
