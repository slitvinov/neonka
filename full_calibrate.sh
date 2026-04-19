#!/bin/sh
# End-to-end per-session calibration + sim generation.
#
# For each session S in 0..61:
#   1. Generate T=55 paired sim with base tables and pooled jumps (-g)
#   2. Measure π_sim(sp, imb), compute d = π_sim/π_real, rescale rates
#   3. Regenerate T=55 sim with calibrated tables
#
# Output: /tmp/neonka/tables/${S}_cal/  and  /tmp/neonka/sim/t55_cal_${S}.raw
set -e

# Pass 1: sim with base tables (use pooled jumps via -g)
echo "pass 1: base sim for all 62 sessions"
for S in $(seq 0 61); do
  ./onestep -D data/train.events -S data/sessions.events.raw -s "$S" \
    -m "/tmp/neonka/tables/$S" -g /tmp/neonka/tables/common \
    -M "/tmp/neonka/hawkes/$S.params" \
    -T 55 -K 100 -j 1 -R 42 > "/tmp/neonka/sim/t55_base_$S.raw" 2>/dev/null
done

# Pass 2: calibrate rates using sim from pass 1
echo "pass 2: calibrate rates for all 62 sessions"
python3 - <<'PY'
import os, glob, shutil
import numpy as np

N_IMB, SP_MAX, EPS = 6, 64, 1e-5

def imb_bin(aN0, bN0, aN1, bN1):
    s = int(aN0)+int(bN0); d = int(aN0)-int(bN0)
    b0 = 1 if s == 0 else (0 if d*5 < -s else (2 if d*5 > s else 1))
    return b0*2 + (1 if aN1 > bN1 else 0)

def load_kv(p):
    if not os.path.exists(p): return {}
    return {int(float(k)): float(v) for k, v in (l.split() for l in open(p) if len(l.split()) == 2)}

def save_kv(p, d):
    with open(p, 'w') as f:
        for k in sorted(d): f.write(f'{k} {d[k]:g}\n')

offs = np.fromfile('data/sessions.raw', dtype=np.int64)
ev = np.memmap('data/train.raw', dtype=np.int32, mode='r').reshape(-1, 49)

for S in range(62):
    r = ev[int(offs[S]):int(offs[S+1])]
    sp_r = (r[:, 0] - r[:, 8]).astype(np.int64)
    imb_r = np.array([imb_bin(a, b, a1, b1) for a, b, a1, b1 in zip(r[:, 32], r[:, 40], r[:, 33], r[:, 41])])
    pi_real = np.zeros((SP_MAX + 1, N_IMB))
    for sp, im in zip(sp_r, imb_r):
        if 0 <= sp <= SP_MAX: pi_real[sp, im] += 1
    pi_real /= pi_real.sum() + 1e-12

    sp_path = f'/tmp/neonka/sim/t55_base_{S}.raw'
    if not os.path.exists(sp_path) or os.path.getsize(sp_path) < 400: continue
    sim = np.fromfile(sp_path, dtype=np.int32).reshape(-1, 49)[1::2]
    sp_s = (sim[:, 0] - sim[:, 8]).astype(np.int64)
    imb_s = np.array([imb_bin(a, b, a1, b1) for a, b, a1, b1 in zip(sim[:, 32], sim[:, 40], sim[:, 33], sim[:, 41])])
    pi_sim = np.zeros((SP_MAX + 1, N_IMB))
    for sp, im in zip(sp_s, imb_s):
        if 0 <= sp <= SP_MAX: pi_sim[sp, im] += 1
    pi_sim /= pi_sim.sum() + 1e-12

    d = np.clip((pi_sim + EPS) / (pi_real + EPS), 0.3, 3.0)

    SRC = f'/tmp/neonka/tables/{S}'
    OUT = f'/tmp/neonka/tables/{S}_cal'
    os.makedirs(OUT, exist_ok=True)

    for ev_name in ('tp', 'tm_q', 'tm_c', 'dp', 'dm'):
        for side in 'ab':
            for im in range(N_IMB):
                src = f'{SRC}/{ev_name}.{side}.imb{im}.rates'
                if not os.path.exists(src): continue
                kv = load_kv(src)
                new_kv = {sp: v * d[min(sp, SP_MAX), im] for sp, v in kv.items()}
                save_kv(f'{OUT}/{ev_name}.{side}.imb{im}.rates', new_kv)
    for f in glob.glob(f'{SRC}/*'):
        name = os.path.basename(f); dst = f'{OUT}/{name}'
        if os.path.exists(dst): continue
        if os.path.islink(f): os.system(f'ln -sf {os.path.realpath(f)} {dst}')
        else: shutil.copy(f, dst)
print('calibrated 62 sessions')
PY

# Pass 3: regenerate T=55 sim with calibrated tables
echo "pass 3: sim with calibrated tables"
for S in $(seq 0 61); do
  ./onestep -D data/train.events -S data/sessions.events.raw -s "$S" \
    -m "/tmp/neonka/tables/${S}_cal" -g /tmp/neonka/tables/common \
    -M "/tmp/neonka/hawkes/$S.params" \
    -T 55 -K 100 -j 1 -R 42 > "/tmp/neonka/sim/t55_h8_$S.raw" 2>/dev/null
done
echo "done — final sim in /tmp/neonka/sim/t55_h8_*.raw"
