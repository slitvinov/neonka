"""Rescale tp rates via detailed balance so the sim's analytical stationary
distribution equals the empirical π_real(sp) of ses45.

For the reduced birth-death chain on sp (verified Markov in earlier analysis):

    π(s)·r_up(s) = π(s+Δ)·r_down(s+Δ)

where:
    r_up(s)     = r_tmc(s) × GAP(s)     (cascade widens by GAP)
    r_down(s)   = r_tp(s) × ⟨dist⟩(s)   × P(dist > 0 | s)

We keep r_tmc and tp jump-distance distribution as-is (data-derived, honest).
The ONLY adjusted quantity is r_tp(s) via a per-sp multiplier c_tp(s) chosen
so that detailed balance holds with π = π_real.

This writes rescaled tp tables to /tmp/neonka/tables/{sid}_db/ and keeps
everything else as symlinks/copies from the original tables dir.
"""
import os, glob, shutil, sys
import numpy as np

SID = int(sys.argv[1]) if len(sys.argv) > 1 else 45
SRC = f'/tmp/neonka/tables/{SID}'
COMMON = '/tmp/neonka/tables/common'
OUT = f'/tmp/neonka/tables/{SID}_db'
os.makedirs(OUT, exist_ok=True)


def load_kv(p):
    if not os.path.exists(p): return {}
    return {int(float(k)): float(v) for k, v in
            (l.split() for l in open(p) if len(l.split()) == 2)}


def save_kv(p, d):
    with open(p, 'w') as f:
        for k in sorted(d):
            f.write(f'{k} {d[k]:g}\n')


def lookup(d, k):
    if not d: return 0.0
    ks = sorted(d); vs = [d[x] for x in ks]
    if k <= ks[0]: return vs[0]
    if k >= ks[-1]: return vs[-1]
    for i in range(1, len(ks)):
        if k <= ks[i]:
            a = (k - ks[i-1]) / (ks[i] - ks[i-1])
            return vs[i-1]*(1-a) + vs[i]*a
    return vs[-1]


# ── empirical π_real(sp), GAP(sp), ⟨dist⟩(sp), P_nz(sp) ─────────────────────
offs = np.fromfile('data/sessions.raw', dtype=np.int64)
r = np.memmap('data/train.raw', dtype=np.int32, mode='r').reshape(-1, 49)[int(offs[SID]):int(offs[SID+1])]
sp_arr = r[:, 0] - r[:, 8]

SP_GRID = list(range(2, 64, 2))
N_SP = len(SP_GRID)

pi_real = np.zeros(N_SP)
for i, sp in enumerate(SP_GRID):
    pi_real[i] = (sp_arr == sp).mean()
# Regularize tiny bins: floor at 1e-4 relative to max, smooth tail
max_p = pi_real.max()
pi_real = np.maximum(pi_real, 1e-4 * max_p)
pi_real /= pi_real.sum()

gap = np.zeros(N_SP)
for i, sp in enumerate(SP_GRID):
    m = (sp_arr == sp) & (r[:, 1] != 0)
    gap[i] = (r[m, 1] - r[m, 0]).mean() if m.sum() > 30 else 2.5


def rate_total(ev, sp):
    tot = 0.0
    for side in ('a', 'b'):
        for im in range(6):
            tot += lookup(load_kv(f'{SRC}/{ev}.{side}.imb{im}.rates'), sp)
    return tot / 6.0


def tp_jump_meanpnz(sp):
    d = load_kv(f'{COMMON}/tp.own.sp{sp}')
    if not d: d = load_kv(f'{COMMON}/tp.own')
    total = sum(d.values())
    if total <= 0: return 0, 0
    nz = {k: v for k, v in d.items() if k > 0}
    p_nz = sum(nz.values()) / total
    mean_nz = sum(k*v for k, v in nz.items()) / max(sum(nz.values()), 1)
    return mean_nz, p_nz


# ── compute c_tp(sp) per sp via detailed balance ───────────────────────────
# Let ρ(s) = π_sim(s) / π_real(s).  If sim over-visits a state, ρ > 1 there.
# At current stationary: π_sim(s)·r_up(s) = π_sim(s+Δ)·r_down(s+Δ)
# We want instead:       π_real(s)·r_up(s) = π_real(s+Δ)·c_tp(s+Δ)·r_down(s+Δ)
# Dividing:              c_tp(s+Δ) = ρ(s+Δ) / ρ(s)
# Intuition: if sim over-concentrates at s+Δ (ρ(s+Δ) > ρ(s)), boost outflow
# from s+Δ by raising tp there; if sim under-visits s+Δ, reduce outflow.

# Need π_sim from analytical chain.  Compute it directly here from current Q.
def rate(ev, sp):
    tot = 0.0
    for side in ('a', 'b'):
        for im in range(6):
            tot += lookup(load_kv(f'{SRC}/{ev}.{side}.imb{im}.rates'), sp)
    return tot / 6.0

def tp_jump_dist(sp):
    d = load_kv(f'{COMMON}/tp.own.sp{sp}')
    if not d: d = load_kv(f'{COMMON}/tp.own')
    total = sum(d.values())
    return {k: v/total for k, v in d.items()} if total > 0 else {}

Q = np.zeros((N_SP, N_SP))
for i, sp in enumerate(SP_GRID):
    r_tp = rate('tp', sp); r_tmc = rate('tm_c', sp)
    jump = tp_jump_dist(sp)
    for d, pd in jump.items():
        if d == 0: Q[i, i] += r_tp * pd
        else:
            sp2 = max(2, sp - d)
            j = min(range(N_SP), key=lambda k: abs(SP_GRID[k] - sp2))
            Q[i, j] += r_tp * pd
    sp2 = sp + int(round(gap[i]))
    if sp2 < max(SP_GRID):
        j = min(range(N_SP), key=lambda k: abs(SP_GRID[k] - sp2))
        Q[i, j] += r_tmc
    Q[i, i] += rate('tm_q', sp) + rate('dp', sp) + rate('dm', sp)

G = Q - np.diag(Q.sum(axis=1))
U, S, Vh = np.linalg.svd(G.T)
pi_sim = np.abs(Vh[-1]); pi_sim /= pi_sim.sum()

# Per-sp over/under ratio
rho = pi_sim / pi_real
rho = np.clip(rho, 1e-3, 1e3)

# Multiplier: c_tp(i+1) = ρ(i+1) / ρ(i).
c_tp = np.ones(N_SP)
for i in range(1, N_SP):
    c_tp[i] = rho[i] / rho[i-1]
    c_tp[i] = max(0.25, min(4.0, c_tp[i]))    # conservative clip

print(f'{"sp":>3}  {"π_real":>8}  {"r_tmc":>8}  {"r_tp":>8}  {"⟨d⟩":>5}  {"P_nz":>5}  {"c_tp":>5}')
for i, sp in enumerate(SP_GRID):
    md, pnz = tp_jump_meanpnz(sp)
    print(f'{sp:>3}  {pi_real[i]:>8.4f}  {rate_total("tm_c",sp):>8.4f}  {rate_total("tp",sp):>8.4f}  '
          f'{md:>5.1f}  {pnz:>5.2f}  {c_tp[i]:>5.2f}')

# ── write rescaled tp tables, copy everything else ─────────────────────────
for f in glob.glob(f'{SRC}/*'):
    name = os.path.basename(f)
    dst = f'{OUT}/{name}'
    if name.startswith('tp.') and 'imb' in name:
        d = load_kv(f)
        new_d = {}
        for sp, v in d.items():
            idx = None
            for i, g in enumerate(SP_GRID):
                if g == sp: idx = i; break
            mult = c_tp[idx] if idx is not None else 1.0
            new_d[sp] = v * mult
        save_kv(dst, new_d)
    else:
        if os.path.islink(f):
            os.system(f'ln -sf {os.path.realpath(f)} {dst}')
        else:
            shutil.copy(f, dst)

print(f'\nwrote rescaled tables to {OUT}')
