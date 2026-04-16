#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { nf = 49, nl = 8, RATE_BASE = 15754 };
enum {
  ASKRATE = 0 * nl,
  BIDRATE = 1 * nl,
  ASKSIZE = 2 * nl,
  BIDSIZE = 3 * nl,
  ASKNC = 4 * nl,
  BIDNC = 5 * nl,
  Y = 6 * nl
};

static char *xstrdup(const char *s) {
  size_t n = strlen(s) + 1;
  char *p = malloc(n);
  if (p == NULL) {
    fprintf(stderr, "convert.c: error: malloc failed in xstrdup\n");
    exit(1);
  }
  memcpy(p, s, n);
  return p;
}

static long float_to_long(const char *field, char **end, int scale) {
  float val;
  long lval;
  double scaled;
  val = strtof(field, end);
  assert(**end == '\0');
  scaled = (double)val * scale;
  lval = lrintf(scaled);
  assert(fabs(scaled - lval) < 1e-6);
  assert(lval >= INT16_MIN && lval <= INT16_MAX);
  return lval;
}

int main(int argc, char **argv) {
  FILE *in, *out;
  char line[16384], *field, *end, *keys[nf];
  int i, n, isnan[nf];
  long long nr, nn, nnan, nans[nf], nviolations, nncsize, zeros[nf], nsort,
      ncross, nncmono, nnancross;
  long lval;
  int16_t ival, ivals[nf];
  long long minv[nf], maxv[nf];
  enum { NBIN = UINT16_MAX + 1, OFFSET = -(int)INT16_MIN };
  long long (*hist)[NBIN] = calloc(nf, sizeof *hist);
  assert(hist != NULL);

  if (argv[1] && strcmp(argv[1], "-h") == 0) {
    puts("usage: convert <input.csv> <output.raw>");
    return 0;
  }
  assert(argv[1]);
  assert(argv[2]);
  assert(argv[3] == NULL);
  if ((in = fopen(argv[1], "r")) == NULL) {
    fprintf(stderr, "convert.c: error: fail to open file '%s'\n", argv[1]);
    exit(1);
  }
  if ((out = fopen(argv[2], "wb")) == NULL) {
    fprintf(stderr, "convert.c: error: fail to open file '%s'\n", argv[2]);
    exit(1);
  }
  if (fgets(line, sizeof line, in) == NULL) {
    fprintf(stderr, "convert.c: error: fail to read file '%s'\n", argv[2]);
    exit(1);
  }
  n = strlen(line);
  assert(n > 0);
  line[n - 1] = '\0';
  field = strtok(line, ",");
  for (i = 0; field != NULL; i++) {
    if (i >= nf) {
      fprintf(stderr, "convert.c: error: i=%d < nf=%d\n", i, nf);
      exit(1);
    }
    keys[i] = xstrdup(field);
    assert(keys[i] != NULL);
    field = strtok(NULL, ",");
  }
  if (i != nf) {
    fprintf(stderr, "convert.c: error: i=%d != nf=%d\n", i, nf);
    exit(1);
  }
  for (i = 0; i < nf; i++) {
    minv[i] = LLONG_MAX;
    maxv[i] = LLONG_MIN;
    nans[i] = 0;
    zeros[i] = 0;
  }
  nr = 0;
  nn = 0;
  nnan = 0;
  nviolations = 0;
  nncsize = 0;
  nsort = 0;
  ncross = 0;
  nncmono = 0;
  nnancross = 0;
  while (fgets(line, sizeof line, in)) {
    n = strlen(line);
    assert(n > 0);
    line[n - 1] = '\0';
    field = strtok(line, ",");
    i = 0;
    while (field != NULL) {
      if (strcmp("NaN", field) == 0) {
        lval = 0;
        isnan[i] = 1;
        nans[i]++;
        nnan++;
      } else {
        isnan[i] = 0;
        assert(field[0] != '0' || field[1] == '\0' || field[1] == '.');
        if (i < ASKSIZE) {
          lval = float_to_long(field, &end, 2) - RATE_BASE;
        } else if (i == Y) {
          lval = float_to_long(field, &end, 4);
        } else {
          lval = strtol(field, &end, 10);
          assert(*end == '\0');
          assert(lval >= INT16_MIN && lval <= INT16_MAX);
        }
        if (lval == 0)
          zeros[i]++;
        if (lval < minv[i])
          minv[i] = lval;
        if (lval > maxv[i])
          maxv[i] = lval;
      }
      ival = (int16_t)lval;
      assert((long)ival == lval);
      ivals[i] = ival;
      hist[i][ival + OFFSET]++;
      fwrite(&ival, sizeof ival, 1, out);
      nn++;
      field = strtok(NULL, ",");
      i++;
    }
    if (i != nf) {
      fprintf(stderr, "convert.c: row %lld has %d fields, expected %d\n",
              nr + 1, i, nf);
      exit(1);
    }
    for (int j = 0; j < nl; j++) {
      int ar = isnan[ASKRATE + j], as = isnan[ASKSIZE + j],
          an = isnan[ASKNC + j];
      int br = isnan[BIDRATE + j], bs = isnan[BIDSIZE + j],
          bn = isnan[BIDNC + j];
      if (ar != as || (ar && an) || (ar && ivals[ASKNC + j] != 0)) {
        if (nnancross < 10)
          printf("NaN cross row %lld: ask level %d: rate=%d size=%d nc=%d "
                 "val=%d\n",
                 nr + 1, j, ar, as, an, ivals[ASKNC + j]);
        nnancross++;
      }
      if (br != bs || (br && bn) || (br && ivals[BIDNC + j] != 0)) {
        if (nnancross < 10)
          printf("NaN cross row %lld: bid level %d: rate=%d size=%d nc=%d "
                 "val=%d\n",
                 nr + 1, j, br, bs, bn, ivals[BIDNC + j]);
        nnancross++;
      }
    }
    for (int g = 0; g < 6; g++) {
      int base = g * nl;
      int saw_nan = 0;
      for (int j = 0; j < nl; j++) {
        if (isnan[base + j])
          saw_nan = 1;
        else if (saw_nan) {
          if (nviolations < 10)
            printf("NaN violation row %lld col %s: "
                   "non-NaN after NaN in group %d\n",
                   nr + 1, keys[base + j], g);
          nviolations++;
        }
      }
    }
    if (ivals[ASKRATE] < ivals[BIDRATE]) {
      if (ncross < 10)
        printf("crossed book row %lld: askRate_0=%d < bidRate_0=%d\n", nr + 1,
               ivals[ASKRATE], ivals[BIDRATE]);
      ncross++;
    }
    for (int j = 1; j < nl; j++) {
      if (!isnan[ASKRATE + j] && !isnan[ASKRATE + j - 1] &&
          ivals[ASKRATE + j] <= ivals[ASKRATE + j - 1]) {
        if (nsort < 10)
          printf("sort violation row %lld: askRate_%d=%d <= askRate_%d=%d\n",
                 nr + 1, j, ivals[ASKRATE + j], j - 1, ivals[ASKRATE + j - 1]);
        nsort++;
      }
      if (!isnan[BIDRATE + j] && !isnan[BIDRATE + j - 1] &&
          ivals[BIDRATE + j] >= ivals[BIDRATE + j - 1]) {
        if (nsort < 10)
          printf("sort violation row %lld: bidRate_%d=%d >= bidRate_%d=%d\n",
                 nr + 1, j, ivals[BIDRATE + j], j - 1, ivals[BIDRATE + j - 1]);
        nsort++;
      }
    }
    for (int j = 0; j < nl; j++) {
      if (ivals[ASKNC + j] > ivals[ASKSIZE + j]) {
        if (nncsize < 10)
          printf("NC>Size row %lld: askNc_%d=%d > askSize_%d=%d\n", nr + 1, j,
                 ivals[ASKNC + j], j, ivals[ASKSIZE + j]);
        nncsize++;
      }
      if (ivals[BIDNC + j] > ivals[BIDSIZE + j]) {
        if (nncsize < 10)
          printf("NC>Size row %lld: bidNc_%d=%d > bidSize_%d=%d\n", nr + 1, j,
                 ivals[BIDNC + j], j, ivals[BIDSIZE + j]);
        nncsize++;
      }
    }
    for (int j = 1; j < nl; j++) {
      if (ivals[ASKNC + j - 1] == 0 && !isnan[ASKNC + j] &&
          ivals[ASKNC + j] != 0) {
        if (nncmono < 10)
          printf("NC mono row %lld: askNc_%d=0 but askNc_%d=%d\n", nr + 1,
                 j - 1, j, ivals[ASKNC + j]);
        nncmono++;
      }
      if (ivals[BIDNC + j - 1] == 0 && !isnan[BIDNC + j] &&
          ivals[BIDNC + j] != 0) {
        if (nncmono < 10)
          printf("NC mono row %lld: bidNc_%d=0 but bidNc_%d=%d\n", nr + 1,
                 j - 1, j, ivals[BIDNC + j]);
        nncmono++;
      }
    }
    nr++;
  }
  if (fclose(in) != 0) {
    fprintf(stderr, "convert.c: error: fclose(in) failed\n");
    exit(1);
  }
  if (fclose(out) != 0) {
    fprintf(stderr, "convert.c: error: fclose(out) failed\n");
    exit(1);
  }
  printf("records/numbers/nans: %lld %lld %lld\n", nr, nn, nnan);
  printf("NaN monotonicity violations: %lld\n", nviolations);
  printf("NC > Size violations: %lld\n", nncsize);
  printf("rate sort violations: %lld\n", nsort);
  printf("crossed book violations: %lld\n", ncross);
  printf("NC zero monotonicity violations: %lld\n", nncmono);
  printf("NaN cross-attribute violations: %lld\n", nnancross);
  printf("col/min/max/nans/zeros/levels\n");
  for (i = 0; i < nf; i++) {
    int levels = 0;
    for (int j = 0; j < NBIN; j++)
      if (hist[i][j])
        levels++;
    if (i > 0 && i % nl == 0)
      printf("\n");
    printf("%-10s %10lld %10lld %8lld %8lld %8d\n", keys[i], minv[i], maxv[i],
           nans[i], zeros[i], levels);
  }
  free(hist);
}
