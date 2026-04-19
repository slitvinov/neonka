"""Calibrate rate tables so the sim's stationary equals the empirical π_real.

Theory: for a continuous-time Markov chain with generator G, scaling all
off-diagonal elements of row i by d_i gives a new chain with stationary
π_new = π_old / d (after normalization).  So to shift the stationary from
π_sim to π_real, set d(s) = π_sim(s) / π_real(s) — i.e. scale every
outgoing rate at state s by that ratio.

In our sim, state = (sp, imb_bin).  Scaling ALL event rates at (sp, imb)
by the same factor d(sp, imb) leaves the within-state event-type mixture
unchanged (tp/tm_c ratios preserved), while modifying the sojourn time.

Inputs:
  π_real(sp, imb)    — joint histogram from training ses45
  π_sim(sp, imb)     — joint histogram from a long(ish) sim (T=55 paired,
                       since our chain-mode sim isn't reliable long-run)

Output: rescaled rate tables written to /tmp/neonka/tables/{sid}_cal/
"""
import os, glob, sys, shutil
import numpy as np

SID = int(sys.argv[1]) if len(sys.argv) > 1 else 45
SRC = f'/tmp/neonka/tables/{SID}'
OUT = f'/tmp/neonka/tables/{SID}_cal'
os.makedirs(OUT, exist_ok=True)

N_IMB = 6


def imb_bin(aN0, bN0, aN1, bN1):
    s = int(aN0) + int(bN0); d = int(aN0) - int(bN0)
    b0 = 1 if s == 0 else (0 if d*5 < -s else (2 if d*5 > s else 1))
    s1 = 1 if aN1 > bN1 else 0
    return b0*2 + s1


def load_kv(p):
    if not os.path.exists(p): return {}
    return {int(float(k)): float(v) for k, v in
            (l.split() for l in open(p) if len(l.split()) == 2)}


def save_kv(p, d):
    with open(p, 'w') as f:
        for k in sorted(d):
            f.write(f'{k} {d[k]:g}\n')


# ── π_real(sp, imb) from training data ──────────────────────────────────────
offs = np.fromfile('data/sessions.raw', dtype=np.int64)
r = np.memmap('data/train.raw', dtype=np.int32, mode='r').reshape(-1, 49)[int(offs[SID]):int(offs[SID+1])]
sp_r = (r[:, 0] - r[:, 8]).astype(np.int64)
imb_r = np.array([imb_bin(a, b, a1, b1) for a, b, a1, b1 in
                  zip(r[:, 32], r[:, 40], r[:, 33], r[:, 41])])

SP_MAX = 64
pi_real = np.zeros((SP_MAX + 1, N_IMB))
for sp, im in zip(sp_r, imb_r):
    if 0 <= sp <= SP_MAX:
        pi_real[sp, im] += 1
pi_real_N = pi_real.sum()
pi_real /= pi_real_N + 1e-12

# ── π_sim(sp, imb) from T=55 paired sim (best proxy for sim stationary) ────
sim_path = '/tmp/neonka/sim/t55_nog.raw'
if not os.path.exists(sim_path):
    print('error: T=55 sim not generated.  Run t55 sim first.', file=sys.stderr)
    sys.exit(1)
sim = np.fromfile(sim_path, dtype=np.int32).reshape(-1, 49)[1::2]
sp_s = (sim[:, 0] - sim[:, 8]).astype(np.int64)
imb_s = np.array([imb_bin(a, b, a1, b1) for a, b, a1, b1 in
                  zip(sim[:, 32], sim[:, 40], sim[:, 33], sim[:, 41])])
pi_sim = np.zeros((SP_MAX + 1, N_IMB))
for sp, im in zip(sp_s, imb_s):
    if 0 <= sp <= SP_MAX:
        pi_sim[sp, im] += 1
pi_sim /= pi_sim.sum() + 1e-12

# ── multiplier d(sp, imb) = π_sim / π_real, with smoothing for low-n bins ──
# Laplace smoothing: add a small count to avoid div by zero in sparse bins.
EPS = 1e-5
d = (pi_sim + EPS) / (pi_real + EPS)
# Clamp to prevent extreme corrections in low-data cells.
d = np.clip(d, 0.3, 3.0)

print(f'{"sp":>3}  {"imb0":>6}  {"imb1":>6}  {"imb2":>6}  {"imb3":>6}  {"imb4":>6}  {"imb5":>6}  <- d(sp,imb)')
for sp in range(2, 50, 2):
    row = f'{sp:>3} '
    for im in range(N_IMB):
        row += f'  {d[sp, im]:>5.2f}'
    print(row)

# ── apply multiplier: scale all rate tables at (sp, imb) by d(sp, imb) ─────
# Event tables keyed by (side, imb) as files; row per sp.  Multiplier depends
# on (sp, imb); same factor applied to all event types at that cell.
EVENTS = [('tp', 'a'), ('tp', 'b'),
          ('tm_q', 'a'), ('tm_q', 'b'),
          ('tm_c', 'a'), ('tm_c', 'b'),
          ('dp', 'a'), ('dp', 'b'),
          ('dm', 'a'), ('dm', 'b')]

for ev, side in EVENTS:
    for im in range(N_IMB):
        src = f'{SRC}/{ev}.{side}.imb{im}.rates'
        if not os.path.exists(src): continue
        kv = load_kv(src)
        new_kv = {sp: v * d[sp if sp <= SP_MAX else SP_MAX, im] for sp, v in kv.items()}
        save_kv(f'{OUT}/{ev}.{side}.imb{im}.rates', new_kv)

# copy everything else (n.imbN, jump tables → symlinks to common)
for f in glob.glob(f'{SRC}/*'):
    name = os.path.basename(f)
    dst = f'{OUT}/{name}'
    if os.path.exists(dst): continue   # already wrote
    if os.path.islink(f):
        os.system(f'ln -sf {os.path.realpath(f)} {dst}')
    else:
        shutil.copy(f, dst)

print(f'\nwrote {OUT}')
