#!/bin/sh
# Compute evidence statistics + generate all plots under sim/figs/.
set -e
cd "$(dirname "$0")/.."
python3 sim/evidence.py
python3 sim/plot.py
python3 sim/params.py
python3 sim/tail_fit.py
python3 sim/kernel_fit.py
echo "done. figures → sim/figs/"
