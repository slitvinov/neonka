#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { nl = 8 };
struct Row {
  int32_t aR[nl];
  int32_t bR[nl];
  int32_t aS[nl];
  int32_t bS[nl];
  int32_t aN[nl];
  int32_t bN[nl];
  int32_t y;
};

int main(int argc, char **argv) {
  int i_arg;
  for (i_arg = 1; i_arg < argc; i_arg++) {
    if (strcmp(argv[i_arg], "-H") == 0) {
      printf("%10s %10s %10s %10s %10s %10s %10s\n", "ntics", "tp", "tm", "dp",
             "dm", "r", "n");
      return 0;
    }
    fprintf(stderr, "events.c: error: unknown flag '%s'\n", argv[i_arg]);
    return 1;
  }
  struct Row prev, cur;
  int64_t tp = 0, tm = 0, dp = 0, dm = 0, r = 0, n = 0, ntics = 0;
  int64_t *tbl[2][2] = {{&tp, &tm}, {&dp, &dm}};
  int32_t dn, ds;
  int i, j;
  while (fread(&prev, sizeof prev, 1, stdin) == 1 &&
         fread(&cur, sizeof cur, 1, stdin) == 1) {
    ntics++;
    if (memcmp(&prev, &cur, sizeof prev) == 0) {
      n++;
      continue;
    }
    i = j = 0;
    while (i < nl && j < nl && prev.aN[i] != 0 && cur.aN[j] != 0) {
      if (cur.aR[j] < prev.aR[i]) {
        (*tbl[j != 0][0])++;
        j++;
      } else if (cur.aR[j] == prev.aR[i]) {
        dn = cur.aN[j] - prev.aN[i];
        ds = cur.aS[j] - prev.aS[i];
        if (dn)
          (*tbl[j != 0][dn < 0])++;
        else if (ds != 0)
          r++;
        i++;
        j++;
      } else {
        (*tbl[i != 0][1])++;
        i++;
      }
    }
  }
  printf("%10lld %10lld %10lld %10lld %10lld %10lld %10lld\n", (long long)ntics,
         (long long)tp, (long long)tm, (long long)dp, (long long)dm,
         (long long)r, (long long)n);
  return 0;
}
