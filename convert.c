#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { nf = 49, nl = 8 };
enum { ASKSIZE = 2 * nl, Y = 6 * nl };

static long float_to_long(char *field, char **end, int scale) {
  float val;
  long lval;
  double scaled;
  val = strtof(field, end);
  assert(**end == '\0');
  scaled = (double)val * scale;
  lval = lrintf(scaled);
  assert(fabs(scaled - lval) < 1e-6);
  return lval;
}

int main(int argc, char **argv) {
  FILE *in = stdin, *out = stdout;
  char line[16384], *field, *end;
  long long nr;
  long lval;
  int32_t ival;
  int i, n;

  if (argc > 1 && strcmp(argv[1], "-h") == 0) {
    fputs("usage: convert < <input.csv> > <output.raw>\n", stderr);
    return 0;
  }
  if (argc != 1) {
    fprintf(stderr, "usage: convert < <input.csv> > <output.raw>\n");
    return 1;
  }
  if (fgets(line, sizeof line, in) == NULL) {
    fprintf(stderr, "convert.c: error: failed to read header\n");
    return 1;
  }
  nr = 0;
  while (fgets(line, sizeof line, in)) {
    n = strlen(line);
    assert(n > 0);
    line[n - 1] = '\0';
    field = strtok(line, ",");
    i = 0;
    while (field != NULL) {
      if (strcmp("NaN", field) == 0) {
        lval = 0;
      } else {
        assert(field[0] != '0' || field[1] == '\0' || field[1] == '.');
        if (i < ASKSIZE || i == Y) {
          lval = float_to_long(field, &end, 4);
        } else {
          lval = strtol(field, &end, 10);
          assert(*end == '\0');
        }
      }
      assert(lval >= INT32_MIN && lval <= INT32_MAX);
      ival = (int32_t)lval;
      assert((long)ival == lval);
      if (fwrite(&ival, sizeof ival, 1, out) != 1) {
        fprintf(stderr, "convert.c: error: fwrite failed\n");
        return 1;
      }
      field = strtok(NULL, ",");
      i++;
    }
    if (i != nf) {
      fprintf(stderr, "convert.c: error: row %lld has %d fields, expected %d\n",
              nr + 1, i, nf);
      return 1;
    }
    nr++;
  }
  return 0;
}
