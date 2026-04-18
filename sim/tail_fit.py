"""Pool ask+bid refill distributions and fit the tail.

For each session, the refill.{a,b}.own histograms count distances from the
newly-surfaced level to the previously-deepest visible level. Under bid/ask
symmetry (verified) these should be iid samples from the same distribution.
Pooling gives 2× N; we then fit parametric tails to see if power-law,
exponential, or log-normal gives a clean extrapolation.
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = f"{HERE}/figs"
TAILDIR = f"{FIGS}/tail"
os.makedirs(TAILDIR, exist_ok=True)

def load_refill(path):
    if not os.path.exists(path): return np.array([]), np.array([])
    arr = np.loadtxt(path, ndmin=2)
    if arr.size == 0: return np.array([]), np.array([])
    return arr[:, 0].astype(int), arr[:, 1].astype(np.int64)

def pool_ab(sid):
    ka, ca = load_refill(f'/tmp/tables{sid}/refill.a.own')
    kb, cb = load_refill(f'/tmp/tables{sid}/refill.b.own')
    keys = sorted(set(ka) | set(kb))
    d = {k: 0 for k in keys}
    for k, c in zip(ka, ca): d[k] += int(c)
    for k, c in zip(kb, cb): d[k] += int(c)
    k_arr = np.array(sorted(d.keys()), dtype=int)
    v_arr = np.array([d[k] for k in k_arr], dtype=np.float64)
    return k_arr, v_arr

def fit_power_law(k, p, k_min=2):
    """Fit log p = α + β log k (power-law) on k >= k_min."""
    mask = (k >= k_min) & (p > 0)
    logk = np.log(k[mask]); logp = np.log(p[mask])
    beta, alpha = np.polyfit(logk, logp, 1)
    return alpha, beta   # p ≈ exp(alpha) * k^beta

def fit_exponential(k, p, k_min=2):
    """Fit log p = α + β k (exponential)."""
    mask = (k >= k_min) & (p > 0)
    k_m, logp = k[mask], np.log(p[mask])
    beta, alpha = np.polyfit(k_m, logp, 1)
    return alpha, beta   # p ≈ exp(alpha) * exp(beta * k)

def fit_lognormal(k, p, k_min=2):
    """Fit log p = α - (log k - μ)^2 / (2σ^2)  via quadratic in log k."""
    mask = (k >= k_min) & (p > 0)
    logk, logp = np.log(k[mask]), np.log(p[mask])
    coef = np.polyfit(logk, logp, 2)   # logp = a*logk^2 + b*logk + c
    a, b, c = coef
    sigma2 = -1.0 / (2 * a) if a < 0 else np.inf
    mu = b * sigma2
    return a, b, c, mu, np.sqrt(max(sigma2, 0))

# --- Per-session plot (62 individual log-log figures) ---
for sid in range(62):
    k, c = pool_ab(sid)
    if len(k) == 0: continue
    tot = c.sum(); p = c / tot

    a_pow, b_pow = fit_power_law(k, p, k_min=4)
    a_exp, b_exp = fit_exponential(k, p, k_min=4)
    a_ln_a, a_ln_b, a_ln_c, mu_ln, sig_ln = fit_lognormal(k, p, k_min=4)

    # Extrapolate beyond observed max to show "unseen region"
    k_max_obs = int(k.max())
    k_grid_obs = np.arange(2, k_max_obs + 1)
    k_grid_ext = np.arange(k_max_obs + 1, max(k_max_obs * 3, 300) + 1)
    k_grid = np.concatenate([k_grid_obs, k_grid_ext])
    p_pow = np.exp(a_pow) * k_grid ** b_pow
    p_exp = np.exp(a_exp) * np.exp(b_exp * k_grid)
    p_ln  = np.exp(a_ln_a * np.log(k_grid)**2 + a_ln_b * np.log(k_grid) + a_ln_c)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(k, p, s=60, color='#1f77b4', label=f'empirical (N={int(tot)}, p(2)={p[0]:.3f})',
               zorder=5, edgecolor='k', linewidth=0.5)
    ax.plot(k_grid, p_pow, 'r-', lw=1.5, label=f'power: $p \\propto k^{{{b_pow:+.2f}}}$')
    ax.plot(k_grid, p_exp, 'g--', lw=1.5, label=f'exp: $p \\propto e^{{{b_exp:+.3f}k}}$')
    ax.plot(k_grid, p_ln, 'm:', lw=1.5,
            label=f'lognormal $\\mu={mu_ln:.2f}$, $\\sigma={sig_ln:.2f}$')
    # Shade extrapolation region
    ax.axvspan(k_max_obs, k_grid_ext[-1], color='gray', alpha=0.12,
               label=f'extrapolated (k>{k_max_obs})')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('refill distance $k$ (ticks)')
    ax.set_ylabel('$P(k)$')
    ax.set_title(f'ses{sid}: ask+bid pooled refill-distance tail fit')
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(alpha=0.3, which='both')
    ax.set_ylim(1e-7, 1.5)
    fig.tight_layout()
    fig.savefig(f'{TAILDIR}/ses{sid:02d}.png', dpi=100)
    plt.close(fig)

print(f"wrote 62 per-session log-log plots to {TAILDIR}/")

# --- Summary table across all 62 sessions ---
print(f"{'ses':>4} {'N':>7} {'P(k=2)':>7} {'tail_frac':>10} "
      f"{'exp_β':>8} {'power_α':>9} {'max_k':>6}")
for sid in range(62):
    k, c = pool_ab(sid)
    if len(k) == 0: continue
    tot = c.sum(); p = c / tot
    p2 = p[0] if k[0] == 2 else 0
    tail = 1 - p2
    a_exp, b_exp = fit_exponential(k, p, k_min=4)
    a_pow, b_pow = fit_power_law(k, p, k_min=4)
    print(f"{sid:>4} {int(tot):>7d} {p2:>7.4f} {tail:>10.4f} "
          f"{b_exp:>+8.4f} {b_pow:>+9.3f} {int(k.max()):>6d}")
