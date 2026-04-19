#!/bin/sh
# T=1 one-step imb baseline sim for one session × K seeds.
set -e
S=$1
K=${K:-50}
D=/tmp/neonka/tables/$S
O=/tmp/neonka/sim/t1_imb_$S
mkdir -p "$O"
for r in $(seq 1 $K); do
  ./session -D data/train.raw -S data/sessions.raw -s "$S" \
    | ./onestep -m "$D" -T 1 -S 100 -W 30 -R $r > "$O/p$r.raw"
done
