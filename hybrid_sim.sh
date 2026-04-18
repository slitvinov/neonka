#!/bin/sh
# Hybrid: imb-bucket baseline + 8-D Hawkes α overlay.
# Usage: sh hybrid_sim.sh <session-id>
set -e
S=$1
K=${K:-50}
D=/tmp/tables$S
P=/tmp/hawkes$S.params
O=/tmp/hy_$S
mkdir -p "$O"
for r in $(seq 1 $K); do
  ./session -D data/train.raw -S data/sessions.raw -s "$S" \
    | ./onestep -m "$D" -T 55 -S 100 -W 30 -M "$P" -Y -R $r > "$O/p$r.raw"
done
