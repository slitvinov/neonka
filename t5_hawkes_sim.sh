#!/bin/sh
# T=5-row horizon Hawkes sim (stationary rate ≈ 0.94 events/unit ≈ 1 row/unit).
# Usage: sh t5_hawkes_sim.sh <session-id>
set -e
S=$1
K=${K:-50}
D=/tmp/tables$S
P=/tmp/hawkes$S.params
O=/tmp/t5_h8_$S.raw
./onestep -D data/train.events -S data/sessions.events.raw -s "$S" \
          -m "$D" -T 5 -K 100 -j "$K" -M "$P" -Z -R 42 > "$O"
