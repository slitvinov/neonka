/* encode.c — LOB encoder (stream)
 * Usage: encode < <input.raw> > <output.lob>
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { nl = 8, CHUNK = 4096 };

struct Row {
  int16_t askRate[nl];
  int16_t bidRate[nl];
  int16_t askSize[nl];
  int16_t bidSize[nl];
  int16_t askNC[nl];
  int16_t bidNC[nl];
  int16_t y;
};

static void w_u8(FILE *f, uint8_t v) { fwrite(&v, 1, 1, f); }
static void w_i16(FILE *f, int16_t v) {
  uint16_t u = (uint16_t)v;
  uint8_t b[2] = {u & 0xff, (u >> 8) & 0xff};
  fwrite(b, 1, 2, f);
}
static void w_i64(FILE *f, int64_t v) {
  uint64_t u = (uint64_t)v;
  uint8_t b[8];
  for (int i = 0; i < 8; i++)
    b[i] = (u >> (8 * i)) & 0xff;
  fwrite(b, 1, 8, f);
}

static int same_side(const int16_t *r1, const int16_t *s1, const int16_t *n1,
                     const int16_t *r0, const int16_t *s0, const int16_t *n0) {
  for (int l = 0; l < nl; l++)
    if (r1[l] != r0[l] || s1[l] != s0[l] || n1[l] != n0[l])
      return 0;
  return 1;
}

static void encode_side(FILE *f, const int16_t *rate1, const int16_t *size1,
                        const int16_t *nc1, const int16_t *rate0,
                        const int16_t *size0, const int16_t *nc0, int asc) {
  int sign = asc ? -1 : 1;
  int j = 0, k = 0, skip = 0;
  while (j < nl && k < nl) {
    int r0 = rate0[j], r1 = rate1[k];
    int s0 = size0[j], s1 = size1[k];
    int n0 = nc0[j], n1 = nc1[k];
    (void)s0;
    (void)n0;
    if (r0 == r1) {
      int ds = s1 - size0[j], dn = n1 - nc0[j];
      if (ds != 0 || dn != 0) {
        w_u8(f, 2);
        w_u8(f, (uint8_t)skip);
        w_i16(f, (int16_t)dn);
        w_i16(f, (int16_t)ds);
        skip = 0;
      } else {
        skip++;
      }
      j++;
      k++;
    } else if ((r0 < r1) == asc) {
      w_u8(f, 1);
      w_u8(f, (uint8_t)skip);
      skip = 0;
      j++;
    } else {
      int ref = k > 0 ? rate1[k - 1] : rate0[j];
      int price = sign * (r1 - ref);
      w_u8(f, 3);
      w_u8(f, (uint8_t)skip);
      w_i16(f, (int16_t)price);
      w_i16(f, (int16_t)n1);
      w_i16(f, (int16_t)s1);
      skip = 0;
      k++;
    }
  }
  if (k < nl) {
    w_u8(f, 4);
    w_i16(f, (int16_t)(nl - k));
    while (k < nl) {
      int r1v = rate1[k], n1v = nc1[k], s1v = size1[k];
      int ref = k > 0 ? rate1[k - 1] : rate0[j - 1];
      int dist = -sign * (r1v - ref);
      w_i16(f, (int16_t)dist);
      w_i16(f, (int16_t)n1v);
      w_i16(f, (int16_t)s1v);
      k++;
    }
  } else {
    w_u8(f, 0);
  }
}

static void encode_lob(const struct Row *rows, int64_t n, int64_t start_tick,
                       FILE *out) {
  w_i64(out, start_tick);
  w_i64(out, n);

  for (int l = 0; l < nl; l++) {
    w_i16(out, rows[0].askRate[l]);
    w_i16(out, rows[0].askNC[l]);
    w_i16(out, rows[0].askSize[l]);
  }
  for (int l = 0; l < nl; l++) {
    w_i16(out, rows[0].bidRate[l]);
    w_i16(out, rows[0].bidNC[l]);
    w_i16(out, rows[0].bidSize[l]);
  }
  w_i16(out, rows[0].y);

  for (int64_t i = 1; i < n; i++) {
    const struct Row *cur = &rows[i];
    const struct Row *prev = &rows[i - 1];
    int ask_same = same_side(cur->askRate, cur->askSize, cur->askNC,
                             prev->askRate, prev->askSize, prev->askNC);
    int bid_same = same_side(cur->bidRate, cur->bidSize, cur->bidNC,
                             prev->bidRate, prev->bidSize, prev->bidNC);
    uint8_t flags = (uint8_t)((!ask_same ? 1 : 0) | (!bid_same ? 2 : 0));
    w_u8(out, flags);
    if (!ask_same)
      encode_side(out, cur->askRate, cur->askSize, cur->askNC, prev->askRate,
                  prev->askSize, prev->askNC, 1);
    if (!bid_same)
      encode_side(out, cur->bidRate, cur->bidSize, cur->bidNC, prev->bidRate,
                  prev->bidSize, prev->bidNC, 0);
    w_i16(out, cur->y);
  }
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  struct Row *buf = malloc((size_t)CHUNK * sizeof *buf);
  if (buf == NULL) {
    fprintf(stderr, "encode.c: error: malloc failed\n");
    return 1;
  }
  int64_t total = 0;
  size_t got;
  while ((got = fread(buf, sizeof *buf, CHUNK, stdin)) > 0) {
    encode_lob(buf, (int64_t)got, total, stdout);
    total += (int64_t)got;
  }
  free(buf);
  return 0;
}
