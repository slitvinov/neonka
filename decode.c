#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { nl = 8 };

struct Row {
  int32_t aR[nl], bR[nl], aS[nl], bS[nl], aN[nl], bN[nl], y;
};

static int r_u8(FILE *f, uint8_t *v) { return fread(v, 1, 1, f) == 1 ? 0 : -1; }
static int r_i16(FILE *f, int16_t *v) {
  uint8_t b[2];
  if (fread(b, 1, 2, f) != 2)
    return -1;
  *v = (int16_t)(b[0] | ((uint16_t)b[1] << 8));
  return 0;
}
static int r_i32(FILE *f, int32_t *v) {
  uint8_t b[4];
  if (fread(b, 1, 4, f) != 4)
    return -1;
  uint32_t u = 0;
  for (int i = 0; i < 4; i++)
    u |= (uint32_t)b[i] << (8 * i);
  *v = (int32_t)u;
  return 0;
}
static int r_i64(FILE *f, int64_t *v) {
  uint8_t b[8];
  if (fread(b, 1, 8, f) != 8)
    return -1;
  uint64_t u = 0;
  for (int i = 0; i < 8; i++)
    u |= (uint64_t)b[i] << (8 * i);
  *v = (int64_t)u;
  return 0;
}

static int decode_side(FILE *f, int32_t *nr, int32_t *nnc, int32_t *ns_,
                       int32_t *or_, int32_t *onc, int32_t *os_, int asc) {
  int sign = asc ? -1 : 1;
  int j = 0, k = 0;

  while (1) {
    uint8_t op;
    if (r_u8(f, &op) != 0)
      return -1;

    if (op == 0) {
      while (k < nl) {
        nr[k] = or_[j];
        nnc[k] = onc[j];
        ns_[k] = os_[j];
        j++;
        k++;
      }
      break;
    } else if (op == 1) {
      uint8_t skip;
      if (r_u8(f, &skip) != 0)
        return -1;
      for (int s = 0; s < (int)skip; s++) {
        nr[k] = or_[j];
        nnc[k] = onc[j];
        ns_[k] = os_[j];
        j++;
        k++;
      }
      j++;
    } else if (op == 2) {
      uint8_t skip;
      int16_t dnc, dsz;
      if (r_u8(f, &skip) != 0 || r_i16(f, &dnc) != 0 || r_i16(f, &dsz) != 0)
        return -1;
      for (int s = 0; s < (int)skip; s++) {
        nr[k] = or_[j];
        nnc[k] = onc[j];
        ns_[k] = os_[j];
        j++;
        k++;
      }
      nr[k] = or_[j];
      nnc[k] = onc[j] + dnc;
      ns_[k] = os_[j] + dsz;
      j++;
      k++;
    } else if (op == 3) {
      uint8_t skip;
      int32_t price;
      int16_t inc, isz;
      if (r_u8(f, &skip) != 0 || r_i32(f, &price) != 0 || r_i16(f, &inc) != 0 ||
          r_i16(f, &isz) != 0)
        return -1;
      for (int s = 0; s < (int)skip; s++) {
        nr[k] = or_[j];
        nnc[k] = onc[j];
        ns_[k] = os_[j];
        j++;
        k++;
      }
      int32_t ref = k > 0 ? nr[k - 1] : or_[j];
      nr[k] = ref + price * sign;
      nnc[k] = inc;
      ns_[k] = isz;
      k++;
    } else if (op == 4) {
      int16_t nrev;
      if (r_i16(f, &nrev) != 0)
        return -1;
      while (k < nl - nrev) {
        nr[k] = or_[j];
        nnc[k] = onc[j];
        ns_[k] = os_[j];
        j++;
        k++;
      }
      for (int ri = 0; ri < (int)nrev; ri++) {
        int32_t dist;
        int16_t rnc, rsz;
        if (r_i32(f, &dist) || r_i16(f, &rnc) || r_i16(f, &rsz))
          return -1;
        int32_t ref = k > 0 ? nr[k - 1] : (j > 0 ? or_[j - 1] : or_[0]);
        nr[k] = ref - dist * sign;
        nnc[k] = rnc;
        ns_[k] = rsz;
        k++;
      }
      break;
    } else {
      fprintf(stderr, "decode.c: error: unknown op %u in decode_side\n", op);
      return -1;
    }
  }
  return 0;
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  FILE *in = stdin;
  FILE *out = stdout;

  struct Row books[2];
  int cur = 0, prv = 1;
  int64_t start_tick, n_ticks;
  int64_t total = 0;

  while (r_i64(in, &start_tick) == 0 && r_i64(in, &n_ticks) == 0) {
    struct Row *b0 = &books[cur];
    memset(b0, 0, sizeof *b0);
    for (int l = 0; l < nl; l++) {
      if (r_i32(in, &b0->aR[l]) || r_i32(in, &b0->aN[l]) ||
          r_i32(in, &b0->aS[l]))
        goto fail;
    }
    for (int l = 0; l < nl; l++) {
      if (r_i32(in, &b0->bR[l]) || r_i32(in, &b0->bN[l]) ||
          r_i32(in, &b0->bS[l]))
        goto fail;
    }
    if (r_i32(in, &b0->y) != 0)
      goto fail;
    fwrite(b0, sizeof(struct Row), 1, out);
    total++;

    for (int64_t i = 1; i < n_ticks; i++) {
      prv = cur;
      cur = 1 - cur;
      struct Row *bc = &books[cur];
      struct Row *bp = &books[prv];

      uint8_t flags;
      if (r_u8(in, &flags) != 0) {
        fprintf(stderr, "decode.c: error: unexpected EOF at tick %lld\n",
                (long long)(total + i));
        goto done;
      }

      if (flags == 0) {
        *bc = *bp;
      } else {
        *bc = *bp;
        if (flags & 1)
          if (decode_side(in, bc->aR, bc->aN, bc->aS, bp->aR,
                          bp->aN, bp->aS, 1) != 0)
            goto fail;
        if (flags & 2)
          if (decode_side(in, bc->bR, bc->bN, bc->bS, bp->bR,
                          bp->bN, bp->bS, 0) != 0)
            goto fail;
      }
      int16_t y16;
      if (r_i16(in, &y16) != 0)
        goto fail;
      bc->y = y16;
      fwrite(bc, sizeof(struct Row), 1, out);
    }
    total += n_ticks - 1;
  }

done:
  return 0;

fail:
  fprintf(stderr, "decode.c: error: stream corrupt\n");
  return 1;
}
