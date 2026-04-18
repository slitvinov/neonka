#!/bin/sh
# Compute evidence statistics + generate all plots under sim/figs/.
set -e
cd "$(dirname "$0")/.."
python3 sim/evidence.py
python3 sim/plot.py
echo "done. figures → sim/figs/"
