#!/bin/sh
# Run onestep with 8-D Hawkes for one session × K seeds.
# Usage: sh hawkes_sim.sh <session-id>
set -e
S=$1
K=${K:-50}
D=/tmp/neonka/tables/$S
P=/tmp/neonka/hawkes/$S.params
O=/tmp/neonka/sim/h8_$S
mkdir -p "$O"
for r in $(seq 1 $K); do
  ./session -D data/train.raw -S data/sessions.raw -s "$S" \
    | ./onestep -m "$D" -T 55 -S 100 -M "$P" -R $r > "$O/p$r.raw"
done
