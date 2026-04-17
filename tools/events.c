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

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  struct Row cur, prev;
  int64_t tp = 0, tm = 0, dp = 0, dm = 0, r = 0;
  int have_prev = 0;
  int k;
  while (fread(&cur, sizeof cur, 1, stdin) == 1) {
    if (have_prev) {
      for (k = 0; k < nl; k++) {
        if (cur.askRate[k] != prev.askRate[k])
          continue;
        int32_t dn = cur.askNC[k] - prev.askNC[k];
        int32_t ds = cur.askSize[k] - prev.askSize[k];
        if (dn > 0) {
          if (k == 0)
            tp++;
          else
            dp++;
        } else if (dn < 0) {
          if (k == 0)
            tm++;
          else
            dm++;
        } else if (ds != 0) {
          r++;
        }
      }
    }
    prev = cur;
    have_prev = 1;
  }
  printf("%lld %lld %lld %lld %lld\n", (long long)tp, (long long)tm,
         (long long)dp, (long long)dm, (long long)r);
  return 0;
}
