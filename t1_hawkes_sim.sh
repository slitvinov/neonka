#!/bin/sh
# T=1 Hawkes + state-dependent μ from imb tables.
# Usage: sh t1_hawkes_sim.sh <session-id>
set -e
S=$1
K=${K:-50}
D=/tmp/neonka/tables/$S
P=/tmp/neonka/hawkes/$S.params
O=/tmp/neonka/sim/t1_h8_$S.raw
./onestep -D data/train.events -S data/sessions.events.raw -s "$S" \
          -m "$D" -T 1 -K 100 -j "$K" -M "$P" -R 42 > "$O"
