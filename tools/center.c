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

enum { OWN, OPP, MID };

int main(int argc, char **argv) {
  int mode = -1;
  int i;
  for (i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
      const char *a = argv[++i];
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
      fprintf(stderr, "usage: center -r own|opp|mid < <input.raw>\n");
      return 1;
    }
  }
  if (mode < 0) {
    fprintf(stderr, "usage: center -r own|opp|mid < <input.raw>\n");
    return 1;
  }
  struct Row r;
  while (fread(&r, sizeof r, 1, stdin) == 1) {
    int j;
    if (mode == OWN) {
      int32_t ref = r.askRate[0];
      for (j = 0; j < nl; j++) {
        r.askRate[j] -= ref;
        r.bidRate[j] -= ref;
      }
    } else if (mode == OPP) {
      int32_t ref = r.bidRate[0];
      for (j = 0; j < nl; j++) {
        r.askRate[j] -= ref;
        r.bidRate[j] -= ref;
      }
    } else {
      int32_t ref = (r.askRate[0] + r.bidRate[0]) / 2;
      for (j = 0; j < nl; j++) {
        r.askRate[j] -= ref;
        r.bidRate[j] -= ref;
      }
    }
    if (fwrite(&r, sizeof r, 1, stdout) != 1) {
      fprintf(stderr, "center.c: error: fwrite failed\n");
      return 1;
    }
  }
  return 0;
}
