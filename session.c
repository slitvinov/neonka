#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

enum { nl = 8 };
struct Row {
  int32_t aR[nl], bR[nl], aS[nl], bS[nl], aN[nl], bN[nl], y;
};

static void help(char *p) {
  fprintf(stderr,
          "usage: %s -D <train.raw> -S <sessions.raw> -s <N>\n"
          "\n"
          "options:\n"
          "  -D file   train data file\n"
          "  -S file   sessions file (int64 boundaries from split)\n"
          "  -s N      session index in [0, #sessions)\n"
          "  -h        help\n",
          p);
}

int main(int argc, char **argv) {
  char *dpath = NULL;
  char *spath = NULL;
  long sid = -1;
  int i;
  for (i = 1; i < argc; i++) {
    char *f = argv[i];
    if (strcmp(f, "-h") == 0) {
      help(argv[0]);
      return 0;
    }
    if (strcmp(f, "-D") == 0 && i + 1 < argc) {
      dpath = argv[++i];
      continue;
    }
    if (strcmp(f, "-S") == 0 && i + 1 < argc) {
      spath = argv[++i];
      continue;
    }
    if (strcmp(f, "-s") == 0 && i + 1 < argc) {
      sid = atol(argv[++i]);
      continue;
    }
    fprintf(stderr, "session.c: error: unknown flag '%s'\n", f);
    help(argv[0]);
    return 1;
  }
  if (dpath == NULL || spath == NULL || sid < 0) {
    fprintf(stderr, "session.c: error: -D, -S, -s are required\n");
    help(argv[0]);
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
  FILE *in = fopen(dpath, "rb");
  if (in == NULL) {
    fprintf(stderr, "session.c: error: fail to open '%s'\n", dpath);
    return 1;
  }
  off_t off = (off_t)beg * (off_t)sizeof(struct Row);
  if (fseeko(in, off, SEEK_SET) != 0) {
    fprintf(stderr, "session.c: error: fseeko failed on '%s'\n", dpath);
    fclose(in);
    return 1;
  }
  struct Row row;
  int64_t r;
  for (r = beg; r < end; r++) {
    if (fread(&row, sizeof row, 1, in) != 1) {
      fprintf(stderr, "session.c: error: short read on '%s'\n", dpath);
      fclose(in);
      return 1;
    }
    if (fwrite(&row, sizeof row, 1, stdout) != 1) {
      fprintf(stderr, "session.c: error: fwrite failed\n");
      fclose(in);
      return 1;
    }
  }
  fclose(in);
  return 0;
}
