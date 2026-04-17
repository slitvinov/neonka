#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { nl = 8 };
struct Row {
  int32_t aR[nl], bR[nl], aS[nl], bS[nl], aN[nl], bN[nl], y;
};

static int diff_ask(int32_t a, int32_t b) { return a - b; }
static int diff_bid(int32_t a, int32_t b) { return b - a; }

static void build(int32_t *oR, int32_t *oN, int32_t *oS, int32_t *pR,
                  int32_t *pN, int32_t *pS, int32_t *cR, int32_t *cN,
                  int32_t *cS, int i, int j) {
  int k, idx;
  for (k = 0; k < nl; k++) {
    if (k < j) {
      oR[k] = cR[k];
      oN[k] = cN[k];
      oS[k] = cS[k];
    } else {
      idx = k - j + i;
      if (idx < nl && pN[idx] != 0) {
        oR[k] = pR[idx];
        oN[k] = pN[idx];
        oS[k] = pS[idx];
      } else {
        oR[k] = cR[k];
        oN[k] = cN[k];
        oS[k] = cS[k];
      }
    }
  }
}

static void emit(int64_t tick, struct Row *r, struct Row *last) {
  if (memcmp(r, last, sizeof *r) == 0)
    return;
  fwrite(&tick, sizeof tick, 1, stdout);
  fwrite(r, sizeof *r, 1, stdout);
  *last = *r;
}

static void walk(int64_t tick, struct Row *out, struct Row *last,
                 int32_t *oR, int32_t *oN, int32_t *oS, int32_t *pR,
                 int32_t *pN, int32_t *pS, int32_t *cR, int32_t *cN,
                 int32_t *cS, int (*diff)(int32_t, int32_t)) {
  int i = 0, j = 0;
  int32_t d, dn, ds, dir, steps, k, saved_cN, saved_cS;
  while (i < nl && j < nl) {
    if (pN[i] == 0 && cN[j] == 0)
      break;
    if (pN[i] == 0)
      d = -1;
    else if (cN[j] == 0)
      d = 1;
    else
      d = diff(cR[j], pR[i]);
    if (d < 0) {
      j++;
      build(oR, oN, oS, pR, pN, pS, cR, cN, cS, i, j);
      emit(tick, out, last);
    } else if (d == 0) {
      dn = cN[j] - pN[i];
      ds = cS[j] - pS[i];
      steps = dn < 0 ? -dn : dn;
      if (steps == 0) {
        i++;
        j++;
        build(oR, oN, oS, pR, pN, pS, cR, cN, cS, i, j);
        emit(tick, out, last);
      } else {
        dir = dn > 0 ? 1 : -1;
        saved_cN = cN[j];
        saved_cS = cS[j];
        for (k = 1; k <= steps; k++) {
          cN[j] = pN[i] + k * dir;
          cS[j] = pS[i] + (int32_t)((int64_t)ds * k / steps);
          build(oR, oN, oS, pR, pN, pS, cR, cN, cS, i + 1, j + 1);
          emit(tick, out, last);
        }
        cN[j] = saved_cN;
        cS[j] = saved_cS;
        i++;
        j++;
      }
    } else {
      i++;
      build(oR, oN, oS, pR, pN, pS, cR, cN, cS, i, j);
      emit(tick, out, last);
    }
  }
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  struct Row prev, cur, out, last;
  int64_t tick = 0;
  while (fread(&prev, sizeof prev, 1, stdin) == 1 &&
         fread(&cur, sizeof cur, 1, stdin) == 1) {
    out = prev;
    memset(&last, 0xff, sizeof last);
    emit(tick, &out, &last);
    walk(tick, &out, &last, out.aR, out.aN, out.aS, prev.aR, prev.aN, prev.aS,
         cur.aR, cur.aN, cur.aS, diff_ask);
    walk(tick, &out, &last, out.bR, out.bN, out.bS, prev.bR, prev.bN, prev.bS,
         cur.bR, cur.bN, cur.bS, diff_bid);
    if (memcmp(&out, &cur, sizeof cur) != 0) {
      out = cur;
      emit(tick, &out, &last);
    }
    tick++;
  }
  return 0;
}
