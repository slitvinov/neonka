#!/bin/sh
# Parallel LOSO: 62 folds × each T via xargs -P.
# Single-threaded per fold (OMP disabled); parallelism comes from xargs only.
# Usage: sh run_ml_loso.sh [T] [--mirror]
set -e
T=${1:-55}
shift
FLAGS="$@"
NP=${NP:-10}

if [ ! -f /tmp/neonka/mlfeat/s0.npz ]; then
  python3 ml_feat.py
fi

# Force each Python fold to be single-threaded. Avoids NP×8 thread contention.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1

TAG=$(echo "$FLAGS" | tr -d ' -' | tr 'a-z' 'A-Z')
[ -n "$TAG" ] && TAG="_$TAG"
echo "T=$T  parallel=$NP  flags=$FLAGS  (OMP=1 per fold)"
OUT=/tmp/neonka/mlloso_T${T}${TAG}.txt
seq 0 61 | xargs -P $NP -I{} python3 ml_fold.py {} $T $FLAGS > $OUT
python3 -c "
import numpy as np
r = {}
for line in open('$OUT'):
    parts = line.split()
    if len(parts) != 3: continue
    ts, T, r2 = int(parts[0]), int(parts[1]), float(parts[2])
    r[ts] = r2
arr = np.array([r.get(ts, 0) for ts in range(62)])
calm = arr[:52]; hot = arr[52:]
print(f'T=$T$TAG  mean={arr.mean():+.3f}%  median={np.median(arr):+.3f}%  n>=0: {(arr>0).sum()}/62')
print(f'  calm (0-51): mean={calm.mean():+.3f}%  median={np.median(calm):+.3f}%')
print(f'  hot  (52-61): mean={hot.mean():+.3f}%  median={np.median(hot):+.3f}%')
print(f'  top 5: ' + ' '.join(f'ses{np.argsort(-arr)[i]}={arr[np.argsort(-arr)[i]]:+.2f}%' for i in range(5)))
"
