#!/bin/sh
# Baseline imb-mode sim for one session × K seeds.
# Usage: sh imb_sim.sh <session-id>
set -e
S=$1
K=${K:-50}
D=/tmp/tables$S
O=/tmp/imb_$S
mkdir -p "$O"
for r in $(seq 1 $K); do
  ./session -D data/train.raw -S data/sessions.raw -s "$S" \
    | ./onestep -m "$D" -T 55 -S 100 -W 30 -R $r > "$O/p$r.raw"
done
