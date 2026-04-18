#!/bin/sh
# Fit 8-D Hawkes for one session, using train.events + sessions.events.raw.
# Usage: sh hawkes_fit.sh <session-id>
set -e
S=$1
./compose -D data/train.events -S data/sessions.events.raw -s "$S" \
  | ./events \
  | ./hawkes -i 500 -b 0.05 -t 1e-7 > "/tmp/hawkes$S.params" 2> "/tmp/hawkes$S.log"
