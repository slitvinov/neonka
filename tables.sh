#!/bin/sh
# Build calibration tables from one session.
# Usage: sh tables.sh [session-id]
set -e
S=${1:-0}
D=${2:-tables}
mkdir -p "$D"

P=$(mktemp /tmp/pairs.XXXXXX)
trap 'rm -f "$P"' EXIT

./session -D data/train.raw -S data/sessions.raw -s "$S" | ./pairs > "$P"

./rates -B sp0_imb < "$P" | awk -v D="$D" '
BEGIN {
  split("tp tm dp dm r", E)
  split("5 6 7 8 9",  A)
  split("10 11 12 13 14", B)
}
function flush(   i, f) {
  for (i=1; i<=5; i++) {
    f = D"/"E[i]".rates"
    print cur, (CA[i]+CB[i])/CN >> f
  }
  f = D"/n.rates"
  print cur, CNE/CN >> f
}
{
  sp=$1; m=$2; n=$3; ne=$4
  for (i=1; i<=5; i++) {
    fa = D"/"E[i]".a.imb"m".rates"
    fb = D"/"E[i]".b.imb"m".rates"
    print sp, $(A[i])/n > fa
    print sp, $(B[i])/n > fb
  }
  fn = D"/n.imb"m".rates"
  print sp, ne/n > fn
  if (sp != cur && NR > 1) { flush(); CN=CNE=0; for (i=1;i<=5;i++) { CA[i]=0; CB[i]=0 } }
  cur=sp; CN+=n; CNE+=ne
  for (i=1; i<=5; i++) { CA[i]+=$(A[i]); CB[i]+=$(B[i]) }
}
END { if (NR > 0) flush() }'

# qr tables: ask-side bucketed by n0_a via raw stream, bid-side bucketed by n0_b via flipped stream
# (rates -B sp0_n0 only buckets by n0_a, so bid-side rates need the flip)
./rates -B sp0_n0 < "$P"           > "$P.qa"
./flip < "$P" | ./rates -B sp0_n0  > "$P.qb"
trap 'rm -f "$P" "$P.qa" "$P.qb"' EXIT

awk -v D="$D" -v side=a '
BEGIN { split("tp tm dp dm r", E); split("5 6 7 8 9", A) }
{ for (i=1; i<=5; i++) { f=D"/qr."E[i]"."side".rates"; print $1, $2, $(A[i])/$3 >> f }
  f=D"/qr.n.rates"; print $1, $2, $4/$3 >> f }
' "$P.qa"
awk -v D="$D" -v side=b '
BEGIN { split("tp tm dp dm r", E); split("5 6 7 8 9", A) }
{ for (i=1; i<=5; i++) { f=D"/qr."E[i]"."side".rates"; print $1, $2, $(A[i])/$3 >> f } }
' "$P.qb"

./tp < "$P" > "$D/tp.own"
./dp < "$P" > "$D/dp.own"
