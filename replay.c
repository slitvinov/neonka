#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { nl = 8 };
struct Row {
  int32_t aR[nl], bR[nl], aS[nl], bS[nl], aN[nl], bN[nl], y;
};
struct Side {
  int32_t *R, *N, *S;
  int (*diff)(int32_t, int32_t);
};

static int diff_ask(int32_t a, int32_t b) { return a - b; }
static int diff_bid(int32_t a, int32_t b) { return b - a; }

static void bind_ask(struct Row *r, struct Side *s) {
  s->R = r->aR;
  s->N = r->aN;
  s->S = r->aS;
  s->diff = diff_ask;
}
static void bind_bid(struct Row *r, struct Side *s) {
  s->R = r->bR;
  s->N = r->bN;
  s->S = r->bS;
  s->diff = diff_bid;
}

static void shift_right(struct Side *s, int i) {
  int k;
  for (k = nl - 1; k > i; k--) {
    s->R[k] = s->R[k - 1];
    s->N[k] = s->N[k - 1];
    s->S[k] = s->S[k - 1];
  }
}
static void shift_left(struct Side *s, struct Side *c, int i) {
  int k;
  for (k = i; k < nl - 1; k++) {
    s->R[k] = s->R[k + 1];
    s->N[k] = s->N[k + 1];
    s->S[k] = s->S[k + 1];
  }
  s->R[nl - 1] = c->R[nl - 1];
  s->N[nl - 1] = c->N[nl - 1];
  s->S[nl - 1] = c->S[nl - 1];
}

static int try_tp(struct Side *s, struct Side *c) {
  if (c->N[0] == 0)
    return 0;
  if (s->N[0] == 0 || s->diff(c->R[0], s->R[0]) < 0) {
    shift_right(s, 0);
    s->R[0] = c->R[0];
    s->N[0] = c->N[0];
    s->S[0] = c->S[0];
    return 1;
  }
  if (s->R[0] == c->R[0] && s->N[0] < c->N[0]) {
    s->N[0] = c->N[0];
    s->S[0] = c->S[0];
    return 1;
  }
  return 0;
}

static int try_tm(struct Side *s, struct Side *c) {
  if (s->N[0] == 0)
    return 0;
  if (c->N[0] == 0 || s->diff(c->R[0], s->R[0]) > 0) {
    shift_left(s, c, 0);
    return 1;
  }
  if (s->R[0] == c->R[0] && s->N[0] > c->N[0]) {
    s->N[0] = c->N[0];
    s->S[0] = c->S[0];
    return 1;
  }
  return 0;
}

static int try_dp(struct Side *s, struct Side *c) {
  int i = 0, j = 0;
  int32_t d;
  while (i < nl && j < nl) {
    if (s->N[i] == 0 && c->N[j] == 0)
      return 0;
    if (s->N[i] == 0)
      d = -1;
    else if (c->N[j] == 0)
      d = 1;
    else
      d = s->diff(c->R[j], s->R[i]);
    if (d < 0) {
      if (i > 0) {
        shift_right(s, i);
        s->R[i] = c->R[j];
        s->N[i] = c->N[j];
        s->S[i] = c->S[j];
        return 1;
      }
      j++;
    } else if (d > 0) {
      i++;
    } else {
      if (i > 0 && s->N[i] < c->N[j]) {
        s->N[i] = c->N[j];
        s->S[i] = c->S[j];
        return 1;
      }
      i++;
      j++;
    }
  }
  return 0;
}

static int try_dm(struct Side *s, struct Side *c) {
  int i = 0, j = 0;
  int32_t d;
  while (i < nl && j < nl) {
    if (s->N[i] == 0 && c->N[j] == 0)
      return 0;
    if (s->N[i] == 0)
      d = -1;
    else if (c->N[j] == 0)
      d = 1;
    else
      d = s->diff(c->R[j], s->R[i]);
    if (d < 0) {
      j++;
    } else if (d > 0) {
      if (i > 0) {
        shift_left(s, c, i);
        return 1;
      }
      i++;
    } else {
      if (i > 0 && s->N[i] > c->N[j]) {
        s->N[i] = c->N[j];
        s->S[i] = c->S[j];
        return 1;
      }
      i++;
      j++;
    }
  }
  return 0;
}

static int try_r(struct Side *s, struct Side *c) {
  int i = 0, j = 0;
  int32_t d;
  while (i < nl && j < nl) {
    if (s->N[i] == 0 && c->N[j] == 0)
      return 0;
    if (s->N[i] == 0)
      d = -1;
    else if (c->N[j] == 0)
      d = 1;
    else
      d = s->diff(c->R[j], s->R[i]);
    if (d < 0)
      j++;
    else if (d > 0)
      i++;
    else {
      if (s->N[i] == c->N[j] && s->S[i] != c->S[j]) {
        s->S[i] = c->S[j];
        return 1;
      }
      i++;
      j++;
    }
  }
  return 0;
}

static int step(struct Side *sa, struct Side *ca, struct Side *sb,
                struct Side *cb) {
  if (try_tp(sa, ca))
    return 1;
  if (try_tp(sb, cb))
    return 1;
  if (try_tm(sa, ca))
    return 1;
  if (try_tm(sb, cb))
    return 1;
  if (try_dp(sa, ca))
    return 1;
  if (try_dp(sb, cb))
    return 1;
  if (try_dm(sa, ca))
    return 1;
  if (try_dm(sb, cb))
    return 1;
  if (try_r(sa, ca))
    return 1;
  if (try_r(sb, cb))
    return 1;
  return 0;
}

static void emit(int64_t tick, struct Row *r) {
  fwrite(&tick, sizeof tick, 1, stdout);
  fwrite(r, sizeof *r, 1, stdout);
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  struct Row prev, cur, s;
  int64_t tick = 0;
  struct Side sa, ca, sb, cb;
  while (fread(&prev, sizeof prev, 1, stdin) == 1 &&
         fread(&cur, sizeof cur, 1, stdin) == 1) {
    emit(tick, &prev);
    s = prev;
    bind_ask(&s, &sa);
    bind_ask(&cur, &ca);
    bind_bid(&s, &sb);
    bind_bid(&cur, &cb);
    while (step(&sa, &ca, &sb, &cb))
      emit(tick, &s);
    if (memcmp(&s, &cur, sizeof s) != 0) {
      s = cur;
      emit(tick, &s);
    }
    tick++;
  }
  return 0;
}
