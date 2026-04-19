"""Analytical stability analysis of the sim's reduced (sp) Markov chain.

Builds a transition matrix Q[sp, sp'] = expected flux per tick FROM sp TO sp',
marginalising over imb_bin and book state using the empirical distribution.

Transition mechanisms per event:
  tp(dist=d):        sp' = sp - d   (if d > 0 and no cross)
  tp(dist=0):        sp' = sp       (queue increment)
  tm_c (cascade):    sp' = sp + GAP(sp)
  tm_q:              sp' = sp (queue decrement on top)
  dp, dm:            sp' = sp (deep levels, no top change)

Analyses:
  1. Stationary distribution π_sim from Q.  Compare to empirical π_real.
  2. Compute drift q_sim(sp) = Σ (sp' − sp) Q[sp, sp']; compare to q_real.
  3. Second-largest eigenvalue: mixing time & slow modes (frozen states).
  4. Identify bins where Q has outgoing flux ≪ real (dead zones).
"""
import os, glob
import numpy as np
from collections import defaultdict


def load_kv(p):
    if not os.path.exists(p): return {}
    return {int(float(k)): float(v) for k, v in
            (l.split() for l in open(p) if len(l.split()) == 2)}


def lookup(d, k):
    if not d: return 0.0
    ks = sorted(d); vs = [d[x] for x in ks]
    if k <= ks[0]: return vs[0]
    if k >= ks[-1]: return vs[-1]
    for i in range(1, len(ks)):
        if k <= ks[i]:
            a = (k - ks[i-1]) / (ks[i] - ks[i-1])
            return vs[i-1] * (1-a) + vs[i] * a
    return vs[-1]


SRC = '/tmp/neonka/tables/45'
COMMON = '/tmp/neonka/tables/common'
SP_GRID = list(range(2, 80, 2))       # 39 sp states; min sp = TICK = 2
N_SP = len(SP_GRID)
SP_TO_IDX = {sp: i for i, sp in enumerate(SP_GRID)}


# ── load empirical π_real(sp) and GAP(sp) ──────────────────────────────────
offs = np.fromfile('data/sessions.raw', dtype=np.int64)
lo, hi = int(offs[45]), int(offs[46])
r = np.memmap('data/train.raw', dtype=np.int32, mode='r').reshape(-1, 49)[lo:hi]
sp_real = r[:, 0] - r[:, 8]
pi_real = np.zeros(N_SP)
for i, sp in enumerate(SP_GRID):
    pi_real[i] = (sp_real == sp).mean()
pi_real /= pi_real.sum() + 1e-30

gap = np.zeros(N_SP)
ar0, ar1 = r[:, 0], r[:, 1]
for i, sp in enumerate(SP_GRID):
    m = (sp_real == sp) & (ar1 != 0)
    gap[i] = (ar1[m] - ar0[m]).mean() if m.sum() > 30 else 2.5


# ── rates at each sp (marginalised over imb) ───────────────────────────────
def rate(ev, sp):
    tot = 0.0
    for s in ('a', 'b'):
        for im in range(6):
            p = f'{SRC}/{ev}.{s}.imb{im}.rates'
            tot += lookup(load_kv(p), sp)
    return tot / 6.0   # assume uniform imb prior (true enough for diagnostic)


def tp_jump_stats(sp):
    d = load_kv(f'{COMMON}/tp.own.sp{sp}')
    if not d: d = load_kv(f'{COMMON}/tp.own')
    if not d: return {}
    total = sum(d.values())
    return {k: v/total for k, v in d.items()}


# ── build Q ────────────────────────────────────────────────────────────────
Q = np.zeros((N_SP, N_SP))    # Q[i, j] = rate sp_i → sp_j per tick
for i, sp in enumerate(SP_GRID):
    r_tp   = rate('tp',   sp)
    r_tmq  = rate('tm_q', sp)
    r_tmc  = rate('tm_c', sp)
    r_dp   = rate('dp',   sp)
    r_dm   = rate('dm',   sp)

    # tp event with dist d: sp' = max(TICK, sp - d).  apply_tp() clamps to
    # opp+TICK when a jump would cross, so minimum post-tp spread is 2 ticks.
    jump = tp_jump_stats(sp)
    for d, pd in jump.items():
        if d == 0:
            Q[i, i] += r_tp * pd
        else:
            sp2 = max(2, sp - d)
            j = min(range(N_SP), key=lambda k: abs(SP_GRID[k] - sp2))
            Q[i, j] += r_tp * pd

    # tm_c causes cascade: sp' = sp + gap(sp)
    sp2 = sp + int(round(gap[i]))
    if sp2 < max(SP_GRID):
        j = min(range(N_SP), key=lambda k: abs(SP_GRID[k] - sp2))
        Q[i, j] += r_tmc

    # queue-only events: sp stays
    Q[i, i] += r_tmq + r_dp + r_dm

# ── stationary π_sim from Q ────────────────────────────────────────────────
# Normalize rows to get transition matrix (P = Q/row_sum, but for continuous-
# time chain, stationary is left eigenvector of generator.  For discrete-tick
# approximation, use Q as instantaneous rates, stationary solves πQ=0 with Σπ=1.
# Equivalent: discrete-time P[i,j] = dt·Q[i,j] + δ[i,j](1 - dt·row_sum) for small dt.
G = Q.copy()
for i in range(N_SP):
    G[i, i] -= G[i].sum()    # G is generator (rows sum to 0)
# π·G = 0 → π is left null vector
U, S, Vh = np.linalg.svd(G.T)
pi_sim = np.abs(Vh[-1])
pi_sim /= pi_sim.sum()

# ── compare π_sim vs π_real ────────────────────────────────────────────────
print(f'{"sp":>3}  {"π_real":>9}  {"π_sim":>9}  {"ratio":>8}  {"drift_Δsp":>12}')
q_sim = np.zeros(N_SP)
for i, sp in enumerate(SP_GRID):
    q_sim[i] = sum((SP_GRID[j] - sp) * Q[i, j] for j in range(N_SP))
for i, sp in enumerate(SP_GRID):
    if pi_real[i] < 1e-5 and pi_sim[i] < 1e-5: continue
    ratio_str = f'{pi_sim[i]/pi_real[i]:>8.2f}' if pi_real[i] > 1e-4 else '     n/a'
    print(f'{sp:>3}  {pi_real[i]:>9.4f}  {pi_sim[i]:>9.4f}  {ratio_str}  {q_sim[i]:>+12.3f}')

# ── slow modes / frozen states: eigenvalues of G ───────────────────────────
eigvals = np.linalg.eigvals(G)
# Sort by magnitude (real part, since G is nearly triangular).  Largest is 0
# (stationary); others are decay rates.
sorted_real = sorted(eigvals.real, reverse=True)
print(f'\nTop 5 eigenvalues of G (generator):')
for i, ev in enumerate(sorted_real[:5]):
    print(f'  λ[{i}] = {ev:+.4f}')
print(f'Mixing time ≈ 1/|λ[1]| = {1/abs(sorted_real[1]):.1f} ticks')
print(f'Slowest non-zero mode = {sorted_real[1]:.4f} (close to 0 → nearly frozen)')

# ── total event rate check ─────────────────────────────────────────────────
total_rate_real = 0.939  # n_events / n_ticks in ses45 (train.events)
avg_rate_sim = sum(Q[i].sum() * pi_sim[i] for i in range(N_SP))
print(f'\nTotal event rate: sim = {avg_rate_sim:.2f} events/tick '
      f'(real ≈ {total_rate_real:.2f}) — ratio {avg_rate_sim/total_rate_real:.2f}')

# Compute E[sp] under both
E_sp_real = sum(SP_GRID[i] * pi_real[i] for i in range(N_SP))
E_sp_sim  = sum(SP_GRID[i] * pi_sim[i]  for i in range(N_SP))
print(f'\nE[sp]: sim={E_sp_sim:.2f}  real={E_sp_real:.2f}  diff={E_sp_sim - E_sp_real:+.2f}')
