#!/bin/sh
# Iterative calibration: keep rescaling rates until sim stationary ≈ π_real.
# Usage: sh iter_calibrate.sh <session-id> [n_iters]
set -e
SID=${1:-45}
N=${2:-5}

# Start from base tables; calibration script reads /tmp/neonka/tables/{SID}_cal
# as input each iteration (after first).
INPUT_DIR=/tmp/neonka/tables/$SID
SIM_OUT=/tmp/neonka/sim/t55_iter_$SID.raw

for i in $(seq 1 $N); do
  OUTDIR=/tmp/neonka/tables/${SID}_iter$i
  rm -rf "$OUTDIR"

  # Calibrate: uses π_sim from SIM_OUT (from previous iter or seed)
  python3 - <<PY
import os, glob, shutil
import numpy as np

SID = $SID
SRC = '$INPUT_DIR'
OUT = '$OUTDIR'
SIM_PATH = '$SIM_OUT'
os.makedirs(OUT, exist_ok=True)

N_IMB = 6; SP_MAX = 64

def imb_bin(aN0, bN0, aN1, bN1):
    s = int(aN0)+int(bN0); d = int(aN0)-int(bN0)
    b0 = 1 if s == 0 else (0 if d*5 < -s else (2 if d*5 > s else 1))
    s1 = 1 if aN1 > bN1 else 0
    return b0*2 + s1

def load_kv(p):
    if not os.path.exists(p): return {}
    return {int(float(k)): float(v) for k, v in (l.split() for l in open(p) if len(l.split()) == 2)}

def save_kv(p, d):
    with open(p, 'w') as f:
        for k in sorted(d): f.write(f'{k} {d[k]:g}\n')

# π_real
offs = np.fromfile('data/sessions.raw', dtype=np.int64)
r = np.memmap('data/train.raw', dtype=np.int32, mode='r').reshape(-1, 49)[int(offs[SID]):int(offs[SID+1])]
sp_r = (r[:, 0] - r[:, 8]).astype(np.int64)
imb_r = np.array([imb_bin(a, b, a1, b1) for a, b, a1, b1 in zip(r[:, 32], r[:, 40], r[:, 33], r[:, 41])])
pi_real = np.zeros((SP_MAX + 1, N_IMB))
for sp, im in zip(sp_r, imb_r):
    if 0 <= sp <= SP_MAX: pi_real[sp, im] += 1
pi_real /= pi_real.sum() + 1e-12

# π_sim from latest sim output
sim = np.fromfile(SIM_PATH, dtype=np.int32).reshape(-1, 49)[1::2]
sp_s = (sim[:, 0] - sim[:, 8]).astype(np.int64)
imb_s = np.array([imb_bin(a, b, a1, b1) for a, b, a1, b1 in zip(sim[:, 32], sim[:, 40], sim[:, 33], sim[:, 41])])
pi_sim = np.zeros((SP_MAX + 1, N_IMB))
for sp, im in zip(sp_s, imb_s):
    if 0 <= sp <= SP_MAX: pi_sim[sp, im] += 1
pi_sim /= pi_sim.sum() + 1e-12

EPS = 1e-5
d = (pi_sim + EPS) / (pi_real + EPS)
d = np.clip(d, 0.3, 3.0)

# Summary stats for this iter
err_l1 = np.abs(pi_sim - pi_real).sum()
e_sp_sim = (sp_s * 1.0).mean(); e_sp_real = (sp_r * 1.0).mean()
print(f'iter $i: ||π_sim − π_real||₁ = {err_l1:.4f}  E[sp]_sim={e_sp_sim:.2f} (target {e_sp_real:.2f})')

for ev in ['tp','tm_q','tm_c','dp','dm']:
    for side in 'ab':
        for im in range(N_IMB):
            src = f'{SRC}/{ev}.{side}.imb{im}.rates'
            if not os.path.exists(src): continue
            kv = load_kv(src)
            new_kv = {sp: v * d[min(sp, SP_MAX), im] for sp, v in kv.items()}
            save_kv(f'{OUT}/{ev}.{side}.imb{im}.rates', new_kv)

for f in glob.glob(f'{SRC}/*'):
    name = os.path.basename(f); dst = f'{OUT}/{name}'
    if os.path.exists(dst): continue
    if os.path.islink(f):
        os.system(f'ln -sf {os.path.realpath(f)} {dst}')
    else:
        shutil.copy(f, dst)
PY

  # Run T=55 sim with the new calibration
  ./onestep -D data/train.events -S data/sessions.events.raw -s "$SID" \
    -m "$OUTDIR" -g /tmp/neonka/tables/common \
    -M "/tmp/neonka/hawkes/$SID.params" \
    -T 55 -K 100 -j 1 -R 42 > "$SIM_OUT"

  # Metrics summary
  ./metrics "$SIM_OUT:even" "$SIM_OUT:odd" 2>&1 | grep -E "spread mean|events/pair" | head -2
  echo "---"

  INPUT_DIR="$OUTDIR"   # next iter starts from this
done
