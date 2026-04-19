#!/bin/sh
# Build calibration tables from one session.
# Usage: sh tables.sh [session-id] [output-dir]
#
# Generates the tables that onestep.c actually loads (see load_tables):
#   tp/tm_q/tm_c/dp/dm × {a,b} × imb{0..5}  — event rates per (sp, imb_bin)
#   n.imb{0..5}                              — "other-event" fractions
#   tp.own, dp.own                           — pooled jump-distance dists
#   tp.own.sp{N}, dp.own.sp{N}               — sp-conditional jump dists
#   refill.{a,b}.own                         — cascade refill distances
#
# Post-processing for full pipeline:
#   python3 pool_tables.py <dir>   # mirror-pool ask↔bid for bid/ask symmetry
#   python3 pool_jumps.py          # pool tp/dp/refill across 62 sessions → shared dir
#
# IMPORTANT (parallel runs): each session writes to its own output-dir; no two
# workers should share an output-dir.  We defensively remove any pre-existing
# symlinks before writing — if pool_jumps.py previously symlinked these files
# to a shared /tmp/tables_common/, writing through the symlink would corrupt
# the shared target.  Re-run pool_jumps.py after tables.sh to refresh the pool.
set -e
S=${1:-0}
D=${2:-tables}
mkdir -p "$D"

# Remove any stale symlinks (from a prior pool_jumps.py run) so we write real
# files here, never through to a shared pooled target.
find "$D" -maxdepth 1 -type l -delete 2>/dev/null || true

P=$(mktemp /tmp/pairs.XXXXXX)
trap 'rm -f "$P"' EXIT

./session -D data/train.raw -S data/sessions.raw -s "$S" | ./pairs > "$P"

# tp/dp/dm and n per (sp, imb).  tm is split into tm_q (queue decrement) and
# tm_c (cascade) by rates_tm_split, which treats them separately — their
# dynamics at wide sp differ by 4× and conflating them drives sim instability.
./rates_tm_split -B sp0_imb < "$P" | awk -v D="$D" '
BEGIN { split("tp tm_q tm_c dp dm", E); split("5 6 7 8 9", A); split("11 12 13 14 15", B) }
{
  sp=$1; m=$2; ntics=$3; n=$4
  if (ntics < 5) next
  for (i=1; i<=5; i++) {
    fa = D"/"E[i]".a.imb"m".rates"
    fb = D"/"E[i]".b.imb"m".rates"
    print sp, $(A[i])/ntics >> fa
    print sp, $(B[i])/ntics >> fb
  }
  fn = D"/n.imb"m".rates"
  print sp, n/ntics >> fn
}'

./tp < "$P" > "$D/tp.own"
./dp < "$P" > "$D/dp.own"

# sp-conditional tp/dp jump distributions: essential at wide sp where the
# pooled global dist has no mass at large jumps needed to close the gap.
./tp_sp -e tp < "$P" | awk -v D="$D" '
{ f = sprintf("%s/tp.own.sp%d", D, $1); print $2, $3 >> f }'
./tp_sp -e dp < "$P" | awk -v D="$D" '
{ f = sprintf("%s/dp.own.sp%d", D, $1); print $2, $3 >> f }'

# Cascade refill distances: refill.c reads row stream, emits "side dist count".
# Split into per-side 2-col tables for onestep (which pools them anyway).
R=$(mktemp /tmp/refill.XXXXXX); trap 'rm -f "$P" "$R"' EXIT
./session -D data/train.raw -S data/sessions.raw -s "$S" | ./refill > "$R"
awk '$1=="a" {print $2, $3}' "$R" > "$D/refill.a.own"
awk '$1=="b" {print $2, $3}' "$R" > "$D/refill.b.own"
