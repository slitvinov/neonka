#!/bin/sh
# One onestep pass for one session (T=1, one timestep).
# Requires /tmp/tables<S> to exist (see tables.sh).
# Usage: sh simulate.sh <session-id>
set -e
S=$1
./session -D data/train.raw -S data/sessions.raw -s "$S" \
  | ./onestep -m "/tmp/neonka/tables/$S" -T 1 -S 100 -R 42 > "/tmp/neonka/sim/sim$S.raw"
