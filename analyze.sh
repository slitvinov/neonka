#!/bin/sh
# Extract (seed, sim_1) pairs from /tmp/t1_h8_<S>.raw and run compare.py per session.
# Usage: sh analyze.sh <session-id>
set -e
S=$1
SRC=/tmp/t1_h8_$S.raw
CMP=/tmp/cmp_src_$S.raw
python3 -c "
import numpy as np
a = np.fromfile('$SRC', dtype=np.int32).reshape(-1, 49)
K = 51                                    # 1 seed + 50 sims per block
n = len(a) // K
b = a.reshape(n, K, 49)
out = np.empty((n*2, 49), dtype=np.int32)
out[0::2] = b[:, 0, :]                    # seeds (real)
out[1::2] = b[:, 1, :]                    # first sim
out.tofile('$CMP')
"
python3 compare.py "$CMP:even" "$CMP:odd" > "/tmp/cmp$S.txt" 2>&1
