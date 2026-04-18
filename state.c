#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { nl = 8 };
struct Row {
  int32_t aR[nl], bR[nl], aS[nl], bS[nl], aN[nl], bN[nl], y;
};

static int32_t v_sp0(struct Row *r) { return r->aR[0] - r->bR[0]; }
static int32_t v_sp1(struct Row *r) { return r->aR[1] - r->bR[1]; }
static int32_t v_sp2(struct Row *r) { return r->aR[2] - r->bR[2]; }
static int32_t v_sp3(struct Row *r) { return r->aR[3] - r->bR[3]; }
static int32_t v_sp4(struct Row *r) { return r->aR[4] - r->bR[4]; }
static int32_t v_sp5(struct Row *r) { return r->aR[5] - r->bR[5]; }
static int32_t v_sp6(struct Row *r) { return r->aR[6] - r->bR[6]; }
static int32_t v_sp7(struct Row *r) { return r->aR[7] - r->bR[7]; }
static int32_t v_aN0(struct Row *r) { return r->aN[0]; }
static int32_t v_bN0(struct Row *r) { return r->bN[0]; }
static int32_t v_aR0(struct Row *r) { return r->aR[0]; }
static int32_t v_bR0(struct Row *r) { return r->bR[0]; }
static int32_t v_y(struct Row *r) { return r->y; }

static struct { char *name; int32_t (*get)(struct Row *); } vars[] = {
    {"sp0", v_sp0}, {"sp1", v_sp1}, {"sp2", v_sp2}, {"sp3", v_sp3},
    {"sp4", v_sp4}, {"sp5", v_sp5}, {"sp6", v_sp6}, {"sp7", v_sp7},
    {"aN0", v_aN0}, {"bN0", v_bN0}, {"aR0", v_aR0}, {"bR0", v_bR0},
    {"y", v_y},     {NULL, NULL}};

enum { OP_EQ, OP_NE, OP_LT, OP_GT, OP_LE, OP_GE };
struct Pred {
  int32_t (*get)(struct Row *);
  int op;
  int32_t val;
};

static int parse_pred(char *expr, struct Pred *p) {
  char *ops[] = {"!=", "<=", ">=", "=", "<", ">", NULL};
  int codes[] = {OP_NE, OP_LE, OP_GE, OP_EQ, OP_LT, OP_GT};
  char *sep = NULL;
  int op = -1, i;
  for (i = 0; ops[i]; i++) {
    sep = strstr(expr, ops[i]);
    if (sep) {
      op = codes[i];
      break;
    }
  }
  if (sep == NULL) {
    fprintf(stderr, "state.c: error: no operator in '%s'\n", expr);
    return -1;
  }
  size_t nlen = (size_t)(sep - expr);
  char *val = sep + strlen(ops[i]);
  for (i = 0; vars[i].name; i++) {
    if (strlen(vars[i].name) == nlen &&
        memcmp(vars[i].name, expr, nlen) == 0) {
      char *end;
      p->get = vars[i].get;
      p->op = op;
      p->val = (int32_t)strtol(val, &end, 10);
      if (*end != '\0') {
        fprintf(stderr, "state.c: error: bad number in '%s'\n", expr);
        return -1;
      }
      return 0;
    }
  }
  fprintf(stderr, "state.c: error: unknown variable in '%s'\n", expr);
  return -1;
}

static int eval(struct Pred *p, struct Row *r) {
  int32_t v = p->get(r);
  switch (p->op) {
  case OP_EQ: return v == p->val;
  case OP_NE: return v != p->val;
  case OP_LT: return v < p->val;
  case OP_GT: return v > p->val;
  case OP_LE: return v <= p->val;
  case OP_GE: return v >= p->val;
  }
  return 0;
}

int main(int argc, char **argv) {
  struct Pred preds[64];
  int npreds = 0, i;
  for (i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
      if (npreds >= 64) {
        fprintf(stderr, "state.c: error: too many -f predicates\n");
        return 1;
      }
      if (parse_pred(argv[++i], &preds[npreds]) != 0)
        return 1;
      npreds++;
      continue;
    }
    fprintf(stderr, "state.c: error: unknown flag '%s'\n", argv[i]);
    return 1;
  }
  struct Row prev, cur;
  while (fread(&prev, sizeof prev, 1, stdin) == 1 &&
         fread(&cur, sizeof cur, 1, stdin) == 1) {
    int pass = 1;
    for (i = 0; i < npreds; i++)
      if (!eval(&preds[i], &prev)) {
        pass = 0;
        break;
      }
    if (!pass)
      continue;
    if (fwrite(&prev, sizeof prev, 1, stdout) != 1 ||
        fwrite(&cur, sizeof cur, 1, stdout) != 1) {
      fprintf(stderr, "state.c: error: fwrite failed\n");
      return 1;
    }
  }
  return 0;
}
