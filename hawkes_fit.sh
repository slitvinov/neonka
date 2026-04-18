#!/bin/sh
# Fit 6-D pooled Hawkes (tp, tm_queue, tm_cascade, dp, dm, hp) for one session.
# Usage: sh hawkes_fit.sh <session-id>
set -e
S=$1
python3 preproc_events.py "$S" \
  | ./hawkes -i 500 -b 0.05 -t 1e-7 > "/tmp/hawkes$S.params" 2> "/tmp/hawkes$S.log"
