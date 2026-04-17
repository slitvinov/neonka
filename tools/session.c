/* session.c — pass through rows belonging to one session
 * Usage: session -S <sessions.raw> -s <N> < <train.raw> > <out.raw>
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

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

int main(int argc, char **argv) {
  const char *spath = NULL;
  long sid = -1;
  int i;
  for (i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-S") == 0 && i + 1 < argc)
      spath = argv[++i];
    else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc)
      sid = atol(argv[++i]);
    else {
      fprintf(stderr,
              "usage: session -S <sessions.raw> -s <N> < <train.raw>\n");
      return 1;
    }
  }
  if (spath == NULL || sid < 0) {
    fprintf(stderr,
            "usage: session -S <sessions.raw> -s <N> < <train.raw>\n");
    return 1;
  }
  FILE *sf = fopen(spath, "rb");
  if (sf == NULL) {
    fprintf(stderr, "session.c: error: fail to open '%s'\n", spath);
    return 1;
  }
  struct stat st;
  if (fstat(fileno(sf), &st) != 0) {
    fprintf(stderr, "session.c: error: fstat failed for '%s'\n", spath);
    fclose(sf);
    return 1;
  }
  long nb = (long)st.st_size / (long)sizeof(int64_t);
  if (nb < 2) {
    fprintf(stderr, "session.c: error: '%s' is too small\n", spath);
    fclose(sf);
    return 1;
  }
  int64_t *store = malloc((size_t)nb * sizeof(int64_t));
  if (store == NULL) {
    fprintf(stderr, "session.c: error: malloc failed\n");
    fclose(sf);
    return 1;
  }
  if (fread(store, sizeof(int64_t), (size_t)nb, sf) != (size_t)nb) {
    fprintf(stderr, "session.c: error: short read on '%s'\n", spath);
    free(store);
    fclose(sf);
    return 1;
  }
  fclose(sf);
  long ns = nb - 1;
  if (sid >= ns) {
    fprintf(stderr, "session.c: error: -s %ld out of range [0, %ld)\n", sid,
            ns);
    free(store);
    return 1;
  }
  int64_t beg = store[sid];
  int64_t end = store[sid + 1];
  free(store);
  struct Row row;
  int64_t r;
  for (r = 0; r < beg; r++) {
    if (fread(&row, sizeof row, 1, stdin) != 1) {
      fprintf(stderr, "session.c: error: short input while skipping\n");
      return 1;
    }
  }
  for (; r < end; r++) {
    if (fread(&row, sizeof row, 1, stdin) != 1) {
      fprintf(stderr, "session.c: error: short input while passing\n");
      return 1;
    }
    if (fwrite(&row, sizeof row, 1, stdout) != 1) {
      fprintf(stderr, "session.c: error: fwrite failed\n");
      return 1;
    }
  }
  return 0;
}
