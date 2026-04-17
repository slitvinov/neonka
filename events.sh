#!/bin/sh
set -e

for s in $(seq 0 62); do
    # printf '%2d ' $s
    ./session -D data/train.raw -S data/sessions.raw -s $s | ./pairs | ./events
done
