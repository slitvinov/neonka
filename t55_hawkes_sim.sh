#!/bin/sh
# T=55 horizon Hawkes sim (matches the y-label: mid[t+55] − mid[t]).
# Usage: sh t55_hawkes_sim.sh <session-id>
set -e
S=$1
K=${K:-50}
D=/tmp/neonka/tables/$S
P=/tmp/neonka/hawkes/$S.params
O=/tmp/neonka/sim/t55_h8_$S.raw
./onestep -D data/train.events -S data/sessions.events.raw -s "$S" \
          -m "$D" -T 55 -K 100 -j "$K" -M "$P" -Z -R 42 > "$O"
