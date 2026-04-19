"""Wu-Rambaldi residual Hawkes fit: λ_c(t) = λ_QR,c(state(t)) + Σ_j α_{c,j} φ_j(t).

QR baseline absorbs the stationary state-conditional rate; α captures only the
self/cross-excitation residual on top.  Fits α via EM (no μ parameter — QR
replaces it).  Cap branching ratio ρ = max_c Σ_j α_{c,j}/β ≤ 0.9.

Writes output in hawkes.c's format (additive sentinel: μ_c = λ_QR,c_avg so
onestep's multiplicative-form λ_stat solve produces M = 1 at stationary).

Usage: python3 fit_wu_residual.py <session_id> [max_iter]
Output: /tmp/neonka/hawkes/<S>_residual.params
"""
import os, sys
import numpy as np

S = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MAX_ITER = int(sys.argv[2]) if len(sys.argv) > 2 else 100
D, BETA, RHO_MAX = 6, 0.05, 0.9
TOL = 1e-6

# ---- pooled type mapping (matches compute_phi_per_state.pooled_type) ----
def pooled_type(t, aN0, bN0):
    if t == 0 or t == 1: return 0
    if t == 2: return 1 if aN0 > 1 else 2
    if t == 3: return 1 if bN0 > 1 else 2
    if t == 4 or t == 5: return 3
    if t == 6 or t == 7: return 4
    if t >= 9: return 5
    return -1

# ---- load rate tables (qr.*.rates) ----
def load_qr_tbl(path, sp_max=256, n_max=32):
    tbl = np.zeros((sp_max, n_max))
    if not os.path.exists(path): return tbl
    for l in open(path):
        p = l.split()
        if len(p) == 3:
            sp, n, r = int(p[0]), int(p[1]), float(p[2])
            if 0 <= sp < sp_max and 0 <= n < n_max:
                tbl[sp, n] = r
    return tbl

tbl_dir = f'/tmp/neonka/tables/{S}_qr_cal'
if not os.path.exists(tbl_dir):
    tbl_dir = f'/tmp/neonka/tables/{S}'
print(f'Loading rate tables from {tbl_dir}')
qr = {}
for ev in ('tp', 'tm', 'dp', 'dm'):
    for side in 'ab':
        qr[f'{ev}.{side}'] = load_qr_tbl(f'{tbl_dir}/qr.{ev}.{side}.rates')

def lam_qr_state(c, sp, an, bn):
    """Per-type rate at state (sp, an, bn), mirrors compute_rates_qr in onestep.c."""
    sp = max(0, min(255, int(sp)))
    an = max(0, min(31, int(an)))
    bn = max(0, min(31, int(bn)))
    tp = qr['tp.a'][sp, an] + qr['tp.b'][sp, bn]
    tm = qr['tm.a'][sp, an] + qr['tm.b'][sp, bn]
    dp = qr['dp.a'][sp, an] + qr['dp.b'][sp, bn]
    dm = qr['dm.a'][sp, an] + qr['dm.b'][sp, bn]
    tm_q = tm if (an > 1 or bn > 1) else 0
    tm_c = tm if (an == 1 or bn == 1) else 0
    if tm_q > 0 and tm_c > 0:
        c_frac = ((an == 1) + (bn == 1)) / 2.0
        tm_c = tm * c_frac
        tm_q = tm * (1.0 - c_frac)
    if c == 0: return tp
    if c == 1: return tm_q
    if c == 2: return tm_c
    if c == 3: return dp
    if c == 4: return dm
    return 0.0  # c == 5 (hp)

# ---- load events ----
REC_SIZE = 54 * 4
offs = np.fromfile('data/sessions.events.raw', dtype=np.int64)
ev_mm = np.memmap('data/train.events', dtype=np.int32, mode='r').reshape(-1, 54)
lo = int(offs[S]) // REC_SIZE
hi = int(offs[S+1]) // REC_SIZE
block = ev_mm[lo:hi]
print(f'Session {S}: {len(block)} rows')

# Build event list: (t, pooled_type, lam_qr_at_state)
ev_t, ev_c, ev_lam = [], [], []
for row in block:
    t_type = int(row[0])
    if t_type == 8: continue  # IDLE marker
    c = pooled_type(t_type, int(row[5+32]), int(row[5+40]))
    if c < 0: continue
    t = int(row[1])
    aR0, bR0 = int(row[5+0]), int(row[5+8])
    sp = aR0 - bR0
    an, bn = int(row[5+32]), int(row[5+40])
    lam = lam_qr_state(c, sp, an, bn)
    ev_t.append(t); ev_c.append(c); ev_lam.append(lam)

N = len(ev_t)
ev_t = np.array(ev_t, dtype=np.int64)
ev_c = np.array(ev_c, dtype=np.int32)
ev_lam = np.array(ev_lam, dtype=np.float64)
T_total = ev_t[-1] - ev_t[0] + 1
counts = np.bincount(ev_c, minlength=D)
print(f'{N} events, T={T_total}, counts={counts}, mean_rate={N/T_total:.4f}')
print(f'  mean λ_QR at events: {ev_lam.mean():.4f} '
      f'(0s: {(ev_lam==0).sum()}/{N})')

# ---- precompute φ at each event time and G_j (compensator) ----
phi_at_event = np.zeros((N, D))
phi = np.zeros(D)
last_t = ev_t[0]
for i in range(N):
    dt = ev_t[i] - last_t
    if dt > 0:
        phi *= np.exp(-BETA * dt)
    phi_at_event[i] = phi
    phi[ev_c[i]] += 1.0
    last_t = ev_t[i]

G = np.zeros(D)
for i in range(N):
    G[ev_c[i]] += (1.0 - np.exp(-BETA * (T_total - (ev_t[i] - ev_t[0])))) / BETA
print(f'G = {G}')

# ---- EM for α ----
alpha = np.full((D, D), 0.005 * BETA)
print(f'\nEM fit with λ_QR baseline, ρ_max={RHO_MAX}:')
ll_prev = -np.inf
for it in range(MAX_ITER):
    # λ per event (using *this event's type row of α*)
    alpha_phi_row_ci = np.einsum('ij,ij->i', alpha[ev_c], phi_at_event)
    lam = ev_lam + alpha_phi_row_ci
    lam = np.maximum(lam, 1e-12)
    ll = np.log(lam).sum() - ev_lam.sum() - (alpha * G[None, :]).sum()

    # Responsibility for α: w_{c_i, j} = α_{c_i, j} φ_j(t_i) / λ_i
    w = alpha[ev_c] * phi_at_event / lam[:, None]  # (N, D)

    alpha_new = np.zeros((D, D))
    for c in range(D):
        mask = ev_c == c
        if G.any() and mask.sum() > 0:
            num = w[mask].sum(axis=0)  # Σ_j
            alpha_new[c] = num / np.maximum(G, 1e-12)

    # ρ-clamp (Gershgorin)
    max_rho = 0
    for c in range(D):
        row_rho = alpha_new[c].sum() / BETA
        max_rho = max(max_rho, row_rho)
        if row_rho > RHO_MAX:
            alpha_new[c] *= RHO_MAX / row_rho

    delta = np.abs(alpha_new - alpha).max()
    alpha = alpha_new
    if (it % 10 == 0) or it < 5:
        print(f'  iter {it:>3}  ll={ll:.2f}  Δll={ll-ll_prev:+.3e}  Δα={delta:.2e}  max_ρ={max_rho:.3f}')
    if it > 0 and abs(ll - ll_prev) < TOL * abs(ll):
        print(f'  converged at iter {it}')
        break
    ll_prev = ll

# ---- output additive-form params: μ_c = 0 signals onestep to use
# rate[c] += Σ_j α_{c,j} φ_j (no linear-system solve, no λ_stat). ----
lam_avg = counts / T_total
print(f'\nλ_avg   = {lam_avg}')
print(f'ρ row   = {alpha.sum(axis=1) / BETA}')

out = f'/tmp/neonka/hawkes/{S}_residual.params'
with open(out, 'w') as f:
    f.write(f'beta 0 {BETA:g}\n')
    for c in range(D):
        f.write(f'mu {c} 0\n')   # 0 → additive mode in onestep
    for c in range(D):
        for j in range(D):
            f.write(f'alpha {c} {j} {alpha[c,j]:g}\n')
print(f'\nWrote {out}')
