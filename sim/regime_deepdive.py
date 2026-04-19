"""D: Regime deep-dive — why do ses52-61 have different kernel structure?

Compare calm (0-51) vs hot (52-61) on:
  - kernel shape (γ_short, γ_long, τ*)
  - event-rate structure (cascade fraction, total rate per type)
  - Hawkes fitted params (ρ, Σμ, α structure)
  - book stationary stats (spread, queue, vol)
  - cross-sectional correlations between regime metrics
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = f"{HERE}/figs"
os.makedirs(FIGS, exist_ok=True)

RECSZ = 216
ev = np.memmap('data/train.events', dtype=np.int32, mode='r').reshape(-1, 54)
offs = np.fromfile('data/sessions.events.raw', dtype=np.int64)

D = 6
LABELS = ['tp','tm_q','tm_c','dp','dm','hp']

def load_params(s):
    p = f'/tmp/hawkes{s}.params'
    if not os.path.exists(p): return None
    beta = 0.05; mu = np.zeros(D); alpha = np.zeros((D, D))
    for l in open(p):
        parts = l.split()
        if not parts: continue
        if parts[0] == 'beta': beta = float(parts[1])
        elif parts[0] == 'mu':
            i = int(parts[1])
            if i < D: mu[i] = float(parts[2])
        elif parts[0] == 'alpha':
            c = int(parts[1]); j = int(parts[2])
            if c < D and j < D: alpha[c, j] = float(parts[3])
    return beta, mu, alpha

def load_acf(s, max_lag=5000):
    lo, hi = int(offs[s])//RECSZ, int(offs[s+1])//RECSZ
    block = ev[lo:hi]
    types = block[:, 0]
    rows = block[:, 1].astype(np.int64) - block[0, 1]
    mask = types < 8
    n_rows = int(rows.max()) + 1
    counts = np.bincount(rows[mask], minlength=n_rows).astype(np.float64)
    x = counts - counts.mean()
    N = len(x)
    M = 1 << int(np.ceil(np.log2(2 * N)))
    fx = np.fft.rfft(x, n=M)
    acf = np.fft.irfft(fx * np.conj(fx), n=M)[:N]
    acf /= acf[0]
    return counts.mean(), acf[:max_lag]

# ── gather per-session stats ─────────────────────────────────────────────────
rows = []
for s in range(62):
    lo, hi = int(offs[s])//RECSZ, int(offs[s+1])//RECSZ
    block = ev[lo:hi]
    types = block[:, 0]
    idles = block[types == 8]
    if len(idles) < 1000: continue

    # fitted Hawkes
    params = load_params(s)
    if params is None: continue
    beta, mu, alpha = params
    try:
        lam = np.linalg.solve(np.eye(D) - alpha/beta, mu)
        rho = float(np.max(np.abs(np.linalg.eigvals(alpha/beta))))
    except np.linalg.LinAlgError:
        lam = np.zeros(D); rho = 0

    # kernel fit (log-log)
    try:
        rate, acf = load_acf(s)
        lg = np.unique(np.round(np.logspace(0, np.log10(len(acf)-1), 80)).astype(int))
        lg = lg[(lg >= 2) & (lg < len(acf))]
        ys = acf[lg]
        m = ys > 1e-6
        ls, ys = lg[m], ys[m]
        g_p, a_p = np.polyfit(np.log(ls), np.log(ys), 1)
        gamma_single = -g_p
    except Exception:
        rate = 0; gamma_single = np.nan

    # book stats
    mid = (idles[:, 5].astype(np.float64) + idles[:, 5+8]) / 2.0
    dmid = np.diff(mid)
    vol = dmid.std()
    sp = (idles[:, 5] - idles[:, 5+8]).astype(np.int64)
    sp_med = float(np.median(sp))
    aN0 = idles[:, 37].astype(np.float64)
    bN0 = idles[:, 45].astype(np.float64)
    imb0_mean = ((aN0 - bN0) / np.maximum(aN0 + bN0, 1)).mean()

    # event-type breakdown (with tm split)
    evm = types < 8
    nonIdle = block[evm]
    tp_cnt = int(((nonIdle[:, 0] == 0) | (nonIdle[:, 0] == 1)).sum())
    tm_all = ((nonIdle[:, 0] == 2) | (nonIdle[:, 0] == 3))
    tm_a = nonIdle[tm_all & (nonIdle[:, 0] == 2)]
    tm_b = nonIdle[tm_all & (nonIdle[:, 0] == 3)]
    tm_q_cnt = int((tm_a[:, 37] > 1).sum()) + int((tm_b[:, 45] > 1).sum())
    tm_c_cnt = int((tm_a[:, 37] == 1).sum()) + int((tm_b[:, 45] == 1).sum())
    dp_cnt = int(((nonIdle[:, 0] == 4) | (nonIdle[:, 0] == 5)).sum())
    dm_cnt = int(((nonIdle[:, 0] == 6) | (nonIdle[:, 0] == 7)).sum())
    total_ev = tp_cnt + tm_q_cnt + tm_c_cnt + dp_cnt + dm_cnt
    cascade_frac = tm_c_cnt / max(tm_q_cnt + tm_c_cnt, 1)

    # (regime set below via data-driven cluster on α[tm_c, tm_c])
    rows.append({
        's': s, 'regime': 0,
        'rho': rho, 'sum_mu': mu.sum(), 'sum_lam': float(lam.sum()),
        'gamma_single': gamma_single, 'rate': rate,
        'vol': vol, 'sp_med': sp_med, 'imb0': imb0_mean,
        'cascade_frac': cascade_frac,
        'n_events': total_ev,
        'alpha_tm_c_tm_c': alpha[2, 2] / beta,    # cascade self-excitation
        'alpha_tm_q_tm_q': alpha[1, 1] / beta,
    })

# ── data-driven regime: hot = α[tm_c, tm_c]/β above 0.1 ────────────────────
THRESHOLD = 0.1
for r in rows: r['regime'] = 1 if r['alpha_tm_c_tm_c'] > THRESHOLD else 0
calm = [r for r in rows if r['regime'] == 0]
hot = [r for r in rows if r['regime'] == 1]
print(f"Data-driven regime assignment (threshold α[tm_c,tm_c]/β > {THRESHOLD}):")
print(f"  calm: {len(calm)} sessions: " + ', '.join(str(r['s']) for r in calm[:3]) +
      f", ... {calm[-1]['s']}" if len(calm) > 3 else "")
print(f"  hot:  {len(hot)} sessions: " + ', '.join(str(r['s']) for r in hot))
print()
keys = ['rho', 'sum_mu', 'sum_lam', 'gamma_single', 'rate', 'vol', 'sp_med',
        'cascade_frac', 'alpha_tm_c_tm_c', 'alpha_tm_q_tm_q', 'imb0']

print(f"{'metric':>22} {'calm mean':>12} {'hot mean':>12} {'ratio (hot/calm)':>18}")
for k in keys:
    c_vals = np.array([r[k] for r in calm if np.isfinite(r[k])])
    h_vals = np.array([r[k] for r in hot  if np.isfinite(r[k])])
    c_mean, h_mean = c_vals.mean(), h_vals.mean()
    ratio = h_mean / c_mean if c_mean != 0 else float('nan')
    print(f"{k:>22} {c_mean:>12.4f} {h_mean:>12.4f} {ratio:>18.3f}")

# ── figure: calm vs hot regime contrasts ─────────────────────────────────────
fig, axs = plt.subplots(3, 3, figsize=(13, 10))
for ax, key, title in zip(
    axs.flat,
    ['vol', 'sp_med', 'cascade_frac',
     'gamma_single', 'rho', 'sum_mu',
     'alpha_tm_c_tm_c', 'alpha_tm_q_tm_q', 'rate'],
    [r'vol(Δmid)', 'median spread', r'cascade fraction (tm_c / tm_all)',
     r'kernel γ (single power-law)', r'branching ρ(α/β)', r'Σμ (baseline)',
     r'α[tm_c, tm_c] / β', r'α[tm_q, tm_q] / β', 'event rate λ']):
    s_ids = [r['s'] for r in rows]
    vals = [r[key] for r in rows]
    colors = ['#1f77b4' if r['regime'] == 0 else '#d62728' for r in rows]
    ax.bar(s_ids, vals, color=colors)
    ax.axvline(51.5, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    ax.set_xlabel('session id', fontsize=8)
fig.suptitle('Calm (blue, 0-51) vs hot (red, 52-61) regime contrast', fontsize=13)
fig.tight_layout()
fig.savefig(f'{FIGS}/23_regime_deepdive.png', dpi=120)
plt.close(fig)

# ── cross-correlation: which regime metrics explain γ, ρ? ───────────────────
print()
print("Cross-session correlations with γ (kernel shape):")
xs = np.array([[r['vol'], r['sp_med'], r['cascade_frac'], r['rate'],
                r['imb0'], r['alpha_tm_c_tm_c']] for r in rows])
ys = np.array([r['gamma_single'] for r in rows])
cols = ['vol', 'sp_med', 'cascade_frac', 'rate', 'imb0', 'α[tm_c,tm_c]']
for i, c in enumerate(cols):
    r = np.corrcoef(xs[:, i], ys)[0, 1]
    print(f"  corr(γ, {c:>18}) = {r:+.3f}")

print()
print("Cross-session correlations with ρ (branching):")
ys = np.array([r['rho'] for r in rows])
for i, c in enumerate(cols):
    r = np.corrcoef(xs[:, i], ys)[0, 1]
    print(f"  corr(ρ, {c:>18}) = {r:+.3f}")

print(f"\nwrote fig: {FIGS}/23_regime_deepdive.png")
