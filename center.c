#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { nl = 8 };
struct Row {
  int32_t aR[nl], bR[nl], aS[nl], bS[nl], aN[nl], bN[nl], y;
};

enum { OWN, OPP, MID };

static void shift(struct Row *r, int32_t ref) {
  int j;
  for (j = 0; j < nl; j++) {
    r->aR[j] -= ref;
    r->bR[j] -= ref;
  }
}

int main(int argc, char **argv) {
  int mode = -1;
  int i;
  for (i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
      char *a = argv[++i];
      if (strcmp(a, "own") == 0)
        mode = OWN;
      else if (strcmp(a, "opp") == 0)
        mode = OPP;
      else if (strcmp(a, "mid") == 0)
        mode = MID;
      else {
        fprintf(stderr, "center.c: error: -r must be own|opp|mid\n");
        return 1;
      }
    } else {
      fprintf(stderr, "usage: center -r own|opp|mid < <pair.raw>\n");
      return 1;
    }
  }
  if (mode < 0) {
    fprintf(stderr, "usage: center -r own|opp|mid < <pair.raw>\n");
    return 1;
  }
  struct Row prev, cur;
  while (fread(&prev, sizeof prev, 1, stdin) == 1 &&
         fread(&cur, sizeof cur, 1, stdin) == 1) {
    int32_t ref;
    if (mode == OWN)
      ref = prev.aR[0];
    else if (mode == OPP)
      ref = prev.bR[0];
    else
      ref = (prev.aR[0] + prev.bR[0]) / 2;
    shift(&prev, ref);
    shift(&cur, ref);
    if (fwrite(&prev, sizeof prev, 1, stdout) != 1 ||
        fwrite(&cur, sizeof cur, 1, stdout) != 1) {
      fprintf(stderr, "center.c: error: fwrite failed\n");
      return 1;
    }
  }
  return 0;
}
