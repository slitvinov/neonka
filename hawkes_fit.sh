#!/bin/sh
# Fit 6-D pooled Hawkes (tp, tm_queue, tm_cascade, dp, dm, hp) for one session.
# Usage: sh hawkes_fit.sh <session-id>
set -e
S=$1
./preproc -D data/train.events -S data/sessions.events.raw -s "$S" \
  | ./hawkes -i 500 -b 0.05 -t 1e-7 > "/tmp/neonka/hawkes/$S.params" 2> "/tmp/neonka/hawkes/$S.log"
