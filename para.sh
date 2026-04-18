#!/bin/sh
# Run a command template in parallel; {} is replaced with each stdin line.
# Usage: seq 0 62 | sh para.sh sh tables.sh {} /tmp/tables{}
exec xargs -P "$(sysctl -n hw.ncpu)" -n 1 -I{} "$@"
