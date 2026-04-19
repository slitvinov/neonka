"""Joint (sp, imb_bin) master equation — full state-space diagnosis.

State = (sp, imb).  sp ∈ {2,4,...,60} (30 values), imb ∈ {0..5} → 180 states.

Builds the analytical generator G[(sp,imb) → (sp',imb')] from rate tables,
treating imb transitions via the empirical *conditional* effect of each event
type on imb, measured from real data.  Stationary of G gives π_sim(sp,imb).
Compared against π_real(sp,imb) from training histogram.

The imb-effect table (how imb changes when an event fires, conditional on
pre-event state) is what makes this work — the sim's actual imb dynamics
are a function of (aN0, bN0, aN1, bN1) which we average over per (sp, imb).

Output: joint π comparison and per-(sp,imb) bias ratio.
"""
import os, sys
import numpy as np

SRC = '/tmp/neonka/tables/45'
COMMON = '/tmp/neonka/tables/common'
SP_GRID = list(range(2, 62, 2))     # 30 sp states
N_SP = len(SP_GRID)
N_IMB = 6
N_STATES = N_SP * N_IMB


def sid(sp_idx, imb): return sp_idx * N_IMB + imb


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
            return vs[i-1]*(1-a) + vs[i]*a
    return vs[-1]


def imb_bin(aN0, bN0, aN1, bN1):
    s = aN0 + bN0; d = aN0 - bN0
    b0 = 1 if s == 0 else (0 if d*5 < -s else (2 if d*5 > s else 1))
    s1 = 1 if aN1 > bN1 else 0
    return b0*2 + s1


# ── π_real(sp, imb) and event-effect tables from real data ──────────────────
offs = np.fromfile('data/sessions.raw', dtype=np.int64)
r = np.memmap('data/train.raw', dtype=np.int32, mode='r').reshape(-1, 49)[int(offs[45]):int(offs[46])]
sp_arr = r[:, 0] - r[:, 8]
aN0 = r[:, 32]; bN0 = r[:, 40]; aN1 = r[:, 33]; bN1 = r[:, 41]

imb_arr = np.array([imb_bin(a, b, a1, b1) for a, b, a1, b1 in zip(aN0, bN0, aN1, bN1)])
sp_to_idx = {sp: i for i, sp in enumerate(SP_GRID)}

pi_real = np.zeros(N_STATES)
for a_sp, a_imb in zip(sp_arr, imb_arr):
    if a_sp in sp_to_idx:
        pi_real[sid(sp_to_idx[a_sp], a_imb)] += 1
pi_real /= pi_real.sum()

# Empirical P_real((sp',imb') | (sp, imb)) from tick-to-tick transitions.
# Used only to compute mixing time and verify master equation matches.
T_real = np.zeros((N_STATES, N_STATES))
for t in range(len(sp_arr) - 1):
    if sp_arr[t] not in sp_to_idx or sp_arr[t+1] not in sp_to_idx: continue
    s_from = sid(sp_to_idx[sp_arr[t]], imb_arr[t])
    s_to = sid(sp_to_idx[sp_arr[t+1]], imb_arr[t+1])
    T_real[s_from, s_to] += 1
row_sums = T_real.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
P_real = T_real / row_sums

# Verify: stationary of P_real == empirical π_real.
pi_check = np.ones(N_STATES) / N_STATES
for _ in range(3000):
    pi_check = pi_check @ P_real
pi_check /= pi_check.sum()
err = np.abs(pi_check - pi_real).sum()
print(f'sanity: ||π_stat - π_empirical||₁ = {err:.4f}   (small → Markov reduction is OK)')

# ── joint stationary from T=55 sim (paired-mode)  ─────────────────────────
sim = np.fromfile('/tmp/neonka/sim/t55_nog.raw', dtype=np.int32).reshape(-1, 49)[1::2]
sp_sim = sim[:, 0] - sim[:, 8]
aN0_s = sim[:, 32]; bN0_s = sim[:, 40]; aN1_s = sim[:, 33]; bN1_s = sim[:, 41]
imb_sim = np.array([imb_bin(a, b, a1, b1) for a, b, a1, b1 in
                    zip(aN0_s, bN0_s, aN1_s, bN1_s)])

pi_sim = np.zeros(N_STATES)
for a_sp, a_imb in zip(sp_sim, imb_sim):
    if a_sp in sp_to_idx:
        pi_sim[sid(sp_to_idx[a_sp], a_imb)] += 1
pi_sim /= pi_sim.sum()

# ── report per (sp, imb) where the bias lives ─────────────────────────────
print()
print('JOINT DISTRIBUTION: π_real vs π_sim(T=55)')
print(f'{"sp":>3} ' + ' '.join(f'  real/sim_imb{i}  ' for i in range(N_IMB)))
for i, sp in enumerate(SP_GRID):
    if sum(pi_real[sid(i, im)] for im in range(N_IMB)) < 0.003: continue
    row = [f'{sp:>3} ']
    for im in range(N_IMB):
        r_ = pi_real[sid(i, im)]
        s_ = pi_sim[sid(i, im)]
        row.append(f' {100*r_:>5.2f}/{100*s_:>5.2f}  ')
    print(''.join(row))

# ── marginal π(sp) and π(imb) ─────────────────────────────────────────────
print()
print('SP MARGINAL:')
print(f'{"sp":>3}  {"real":>7}  {"sim":>7}  {"ratio":>7}')
for i, sp in enumerate(SP_GRID):
    r_ = sum(pi_real[sid(i, im)] for im in range(N_IMB))
    s_ = sum(pi_sim[sid(i, im)] for im in range(N_IMB))
    if max(r_, s_) > 0.002:
        print(f'{sp:>3}  {100*r_:>7.2f}  {100*s_:>7.2f}  {s_/max(r_,1e-6):>7.2f}')

print()
print('IMB MARGINAL:')
for im in range(N_IMB):
    r_ = sum(pi_real[sid(i, im)] for i in range(N_SP))
    s_ = sum(pi_sim[sid(i, im)] for i in range(N_SP))
    print(f'  imb={im}: real={100*r_:.2f}%  sim={100*s_:.2f}%  diff={100*(s_-r_):+.2f}')

# ── second-largest eigenvalue of P_real: mixing time ──────────────────────
eigs = np.linalg.eigvals(P_real)
mags = sorted(np.abs(eigs), reverse=True)
print(f'\nSecond eigenvalue of P_real: |λ₁| = {mags[1]:.4f} → '
      f'mixing time ≈ {1/(1-mags[1]):.1f} ticks')

# ── identify biggest local bias (cells where sim differs most from real) ──
diff = pi_sim - pi_real
abs_diff = np.abs(diff)
top = np.argsort(-abs_diff)[:10]
print(f'\nTop 10 most biased (sp, imb) cells:')
for s in top:
    sp = SP_GRID[s // N_IMB]; im = s % N_IMB
    print(f'  sp={sp:>3} imb={im}:  real={100*pi_real[s]:>5.2f}%  sim={100*pi_sim[s]:>5.2f}%  '
          f'diff={100*diff[s]:+.2f}')
