/* center.c — shift rates so that a chosen reference sits at 0
 * Usage: center -r own|opp|mid < <input.raw> > <output.raw>
 *   own : subtract askRate[0]           (rate unit unchanged: 1/2 tick)
 *   opp : subtract bidRate[0]           (rate unit unchanged: 1/2 tick)
 *   mid : 2*rate - (askRate[0]+bidRate[0])   (rate unit halves: 1/4 tick,
 *         matching y's 1/4-tick scale so y is left unchanged)
 * Size and NC are unchanged.  Breaks absolute-price invariants; downstream
 * reads this as distance-from-reference.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
      int ref = r.askRate[0];
      for (j = 0; j < nl; j++) {
        r.askRate[j] = (int16_t)(r.askRate[j] - ref);
        r.bidRate[j] = (int16_t)(r.bidRate[j] - ref);
      }
    } else if (mode == OPP) {
      int ref = r.bidRate[0];
      for (j = 0; j < nl; j++) {
        r.askRate[j] = (int16_t)(r.askRate[j] - ref);
        r.bidRate[j] = (int16_t)(r.bidRate[j] - ref);
      }
    } else {
      int ref = r.askRate[0] + r.bidRate[0];
      for (j = 0; j < nl; j++) {
        r.askRate[j] = (int16_t)(2 * r.askRate[j] - ref);
        r.bidRate[j] = (int16_t)(2 * r.bidRate[j] - ref);
      }
    }
    if (fwrite(&r, sizeof r, 1, stdout) != 1) {
      fprintf(stderr, "center.c: error: fwrite failed\n");
      return 1;
    }
  }
  return 0;
}
