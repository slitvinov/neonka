"""Test 1 extended: kernel form diagnostic + 2-exponential mixture fit.

Compare three kernel hypotheses for the event-rate ACF:
  - single exponential exp(-β·τ):                    Bacry 2015 / current sim
  - power-law         τ^(-γ):                       Bacry-Jaisson 2016
  - 2-exponential mix c1·β1·e^-β1τ + c2·β2·e^-β2τ:  approximate power-law

Produces sim/figs/19_kernel_acf.png (log-linear + log-log panels) and
sim/figs/20_kernel_mix_fit.png (2-exp mixture vs single-exp vs power-law).
Prints fit R² and BIC-like per-model score.
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = f"{HERE}/figs"
os.makedirs(FIGS, exist_ok=True)

RECSZ = 216
ev = np.memmap('data/train.events', dtype=np.int32, mode='r').reshape(-1, 54)
offs = np.fromfile('data/sessions.events.raw', dtype=np.int64)

def load_acf(s, max_lag=5000):
    """Event-count ACF per row for session s."""
    lo, hi = int(offs[s])//RECSZ, int(offs[s+1])//RECSZ
    block = ev[lo:hi]
    types = block[:, 0]
    rows_global = block[:, 1].astype(np.int64)
    rows = rows_global - rows_global.min()
    ev_mask = types < 8
    n_rows = int(rows.max()) + 1
    counts = np.bincount(rows[ev_mask], minlength=n_rows).astype(np.float64)
    x = counts - counts.mean()
    N = len(x)
    M = 1 << int(np.ceil(np.log2(2 * N)))
    fx = np.fft.rfft(x, n=M)
    acf = np.fft.irfft(fx * np.conj(fx), n=M)[:N]
    acf /= acf[0]
    return counts.mean(), acf[:max_lag]

# Use ses45 (middle-regime, well-calibrated)
s_test = 45
mean_rate, acf = load_acf(s_test)
print(f"ses{s_test}: {len(acf)} lags, mean events/row = {mean_rate:.4f}")

# Log-spaced lag grid for fitting
lags = np.unique(np.round(np.logspace(0, np.log10(len(acf)-1), 80)).astype(int))
lags = lags[(lags >= 2) & (lags < len(acf))]
y = acf[lags]
pos = y > 1e-6
lags, y = lags[pos], y[pos]
logL = np.log(lags); logY = np.log(y)

# ---- Model 1: single exponential C·exp(-β·τ) ----
# log Y = log C - β·τ
b_exp, a_exp = np.polyfit(lags, logY, 1)
C_exp, beta_exp = np.exp(a_exp), -b_exp
pred_exp = C_exp * np.exp(-beta_exp * lags)
ss_res_exp = ((logY - (a_exp + b_exp*lags))**2).sum()
ss_tot = ((logY - logY.mean())**2).sum()
r2_exp = 1 - ss_res_exp/ss_tot

# ---- Model 2: power-law C·τ^(-γ) ----
g_pow, a_pow = np.polyfit(logL, logY, 1)
C_pow, gamma_pow = np.exp(a_pow), -g_pow
pred_pow = C_pow * lags**(-gamma_pow)
ss_res_pow = ((logY - (a_pow + g_pow*logL))**2).sum()
r2_pow = 1 - ss_res_pow/ss_tot

# ---- Model 3: 2-exponential mixture ----
def mix2(tau, c1, b1, c2, b2):
    return c1 * np.exp(-b1 * tau) + c2 * np.exp(-b2 * tau)

try:
    # Good initialization: fast-β=0.05 (current sim), slow-β=0.005
    p0 = [acf[1]*0.7, 0.05, acf[1]*0.3, 0.005]
    bounds = ([0, 1e-5, 0, 1e-5], [1, 5, 1, 5])
    popt, _ = curve_fit(mix2, lags, y, p0=p0, bounds=bounds, maxfev=10000)
    c1, b1, c2, b2 = popt
    pred_mix = mix2(lags, *popt)
    # Sort components: faster first
    if b1 < b2:
        c1, b1, c2, b2 = c2, b2, c1, b1
    ss_res_mix = ((np.log(pred_mix) - logY)**2).sum()
    r2_mix = 1 - ss_res_mix/ss_tot
except Exception as e:
    print(f"mix fit failed: {e}")
    c1 = b1 = c2 = b2 = 0
    pred_mix = np.zeros_like(lags, dtype=float)
    r2_mix = 0

print(f"\n=== Kernel fits on ses{s_test} event-rate ACF ===")
print(f"  single-exp:   y = {C_exp:.3f}·exp(-{beta_exp:.5f}·τ)         R² = {r2_exp:.4f}")
print(f"  power-law:    y = {C_pow:.3f}·τ^(-{gamma_pow:.3f})               R² = {r2_pow:.4f}")
print(f"  2-exp mix:    y = {c1:.3f}·e^(-{b1:.4f}τ) + {c2:.3f}·e^(-{b2:.4f}τ)  R² = {r2_mix:.4f}")

# ---- Figure 1: two-panel diagnostic (log-linear + log-log) ----
fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
tau_smooth = np.logspace(0, np.log10(lags.max()), 300)
y_lo = max(y.min() * 0.5, 1e-4)
y_hi = min(y.max() * 2.0, 1.5)
def clip(yy): return np.maximum(yy, y_lo)

ax = axs[0]
ax.semilogy(lags, clip(y), 'o', ms=4, color="#1f77b4", label="empirical")
ax.semilogy(tau_smooth, clip(C_exp * np.exp(-beta_exp * tau_smooth)),
            'r-', lw=1.5, label=f"single-exp β={beta_exp:.4f}  R²={r2_exp:.3f}")
ax.semilogy(tau_smooth, clip(0.263 * np.exp(-0.05 * tau_smooth)),
            'r:', lw=1.5, label="our sim β=0.05")
ax.semilogy(tau_smooth, clip(c1*np.exp(-b1*tau_smooth) + c2*np.exp(-b2*tau_smooth)),
            'b-', lw=1.5, label=f"2-exp mix  R²={r2_mix:.3f}")
ax.set_xlabel('lag τ (rows)')
ax.set_ylabel('ACF(τ)')
ax.set_title(f'Log-linear: pure exponential = straight line')
ax.set_ylim(y_lo, y_hi)
ax.legend(fontsize=10)
ax.grid(alpha=0.25)

ax = axs[1]
ax.loglog(lags, clip(y), 'o', ms=4, color="#1f77b4", label="empirical")
ax.loglog(tau_smooth, clip(C_pow * tau_smooth**(-gamma_pow)),
          'g-', lw=1.5, label=f"power-law γ={gamma_pow:.3f}  R²={r2_pow:.3f}")
ax.loglog(tau_smooth, clip(C_exp * np.exp(-beta_exp * tau_smooth)),
          'r--', lw=1, label=f"single-exp (poor fit)")
ax.loglog(tau_smooth, clip(c1*np.exp(-b1*tau_smooth) + c2*np.exp(-b2*tau_smooth)),
          'b-', lw=1.5, label=f"2-exp mix  R²={r2_mix:.3f}")
ax.set_xlabel('lag τ (rows)')
ax.set_ylabel('ACF(τ)')
ax.set_title('Log-log: pure power-law = straight line')
ax.set_ylim(y_lo, y_hi)
ax.legend(fontsize=10)
ax.grid(alpha=0.25, which="both")

fig.suptitle(f'Test 1: kernel form from event-rate ACF (ses{s_test})', fontsize=13)
fig.tight_layout()
fig.savefig(f'{FIGS}/19_kernel_acf.png', dpi=120)
plt.close(fig)

# ---- Figure 2: kernel form across 4 representative sessions ----
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
SES_SHOW = [0, 30, 45, 56]
for ax, sid in zip(axs.flat, SES_SHOW):
    rate, acf_s = load_acf(sid)
    lags_s = np.unique(np.round(np.logspace(0, np.log10(len(acf_s)-1), 80)).astype(int))
    lags_s = lags_s[(lags_s >= 2) & (lags_s < len(acf_s))]
    y_s = acf_s[lags_s]
    mask_s = y_s > 1e-6
    ls, ys = lags_s[mask_s], y_s[mask_s]
    if len(ls) < 3: continue
    # Fits
    b_e, a_e = np.polyfit(ls, np.log(ys), 1)
    g_p, a_p = np.polyfit(np.log(ls), np.log(ys), 1)
    r2_e = 1 - ((np.log(ys) - (a_e + b_e*ls))**2).sum() / ((np.log(ys) - np.log(ys).mean())**2).sum()
    r2_p = 1 - ((np.log(ys) - (a_p + g_p*np.log(ls)))**2).sum() / ((np.log(ys) - np.log(ys).mean())**2).sum()
    ax.loglog(ls, ys, 'o', ms=4, color="#1f77b4", label="data")
    tau_s = np.logspace(0, np.log10(ls.max()), 200)
    ax.loglog(tau_s, np.exp(a_e)*np.exp(b_e*tau_s), 'r--', lw=1.2, label=f"exp R²={r2_e:.2f}")
    ax.loglog(tau_s, np.exp(a_p)*tau_s**g_p, 'g-', lw=1.5, label=f"power R²={r2_p:.2f}")
    ax.set_title(f'ses{sid}: power-law γ={-g_p:.2f}, λ_bar={rate:.3f}')
    ax.set_xlabel('lag τ'); ax.set_ylabel('ACF')
    ax.legend(fontsize=9); ax.grid(alpha=0.25, which="both")
fig.suptitle("Kernel form per session — power-law fits consistently better", fontsize=13)
fig.tight_layout()
fig.savefig(f'{FIGS}/20_kernel_per_session.png', dpi=120)
plt.close(fig)

# ---- Figure 3: residuals from single-exp β=0.05 (our current sim) ----
# Restrict to lags where the sim kernel is still numerically meaningful
# (β=0.05·τ < 30, beyond which the exp is <1e-13 and ratios are noise).
fig, ax = plt.subplots(figsize=(9, 5.5))
MAX_SAFE_TAU = 600               # 0.05·600 = 30; exp(-30) ≈ 1e-13
sel = lags <= MAX_SAFE_TAU
lags_r = lags[sel]; y_r = y[sel]
residual = y_r / (0.263 * np.exp(-0.05 * lags_r))
ax.loglog(lags_r, residual, 'o-', color="#d62728", ms=4, lw=0.8,
          label='empirical / our-sim-kernel')
ax.axhline(1.0, color='k', lw=0.5, label='match (ratio = 1)')
ax.set_xlabel('lag τ (rows)')
ax.set_ylabel(r'ACF_empirical(τ) / [0.263·exp(−0.05·τ)]')
ax.set_title(r'Residual: real ACF / our sim\'s kernel — blows up by $\gtrsim 10^8$ at τ=500')
ax.grid(alpha=0.3, which="both")
ax.set_ylim(1, 1e10)
ax.legend()
ann_idx = np.searchsorted(lags_r, 100)
if ann_idx < len(lags_r):
    ax.annotate(f'τ=100: {residual[ann_idx]:.0f}×\nreal correlation\nis {residual[ann_idx]:.0f}× stronger',
                xy=(lags_r[ann_idx], residual[ann_idx]),
                xytext=(20, residual[ann_idx] * 0.3),
                fontsize=10, color='#444',
                arrowprops=dict(arrowstyle='->', color='#888'))
fig.tight_layout()
fig.savefig(f'{FIGS}/21_kernel_residual.png', dpi=120)
plt.close(fig)

# ---- Per-session kernel ACF + fit, one PNG per session ----
KDIR = f"{FIGS}/kernel"
os.makedirs(KDIR, exist_ok=True)
print(f"generating per-session kernel plots → {KDIR}/ ...")
summary = []
for sid in range(62):
    try:
        rate, acf_s = load_acf(sid)
    except Exception as e:
        print(f"  ses{sid}: {e}"); continue
    lg = np.unique(np.round(np.logspace(0, np.log10(len(acf_s)-1), 80)).astype(int))
    lg = lg[(lg >= 2) & (lg < len(acf_s))]
    ys = acf_s[lg]
    msk = ys > 1e-6
    if msk.sum() < 3: continue
    ls, ys = lg[msk], ys[msk]

    b_e, a_e = np.polyfit(ls, np.log(ys), 1)
    g_p, a_p = np.polyfit(np.log(ls), np.log(ys), 1)
    pred_e = np.exp(a_e + b_e*ls)
    pred_p = np.exp(a_p + g_p*np.log(ls))
    sstot = ((np.log(ys) - np.log(ys).mean())**2).sum()
    r2_e = 1 - ((np.log(ys) - np.log(pred_e))**2).sum() / sstot
    r2_p = 1 - ((np.log(ys) - np.log(pred_p))**2).sum() / sstot

    # Broken power-law: search breakpoint in a reasonable range, prefer τ* with
    # γ_short ≥ γ_long (short lags decay at least as fast — physical expectation).
    best = (r2_p, None, None, None, None, None, None)
    logL, logY = np.log(ls), np.log(ys)
    MIN_SIDE = 6                       # min points per regime
    # Only consider breakpoints in interior: τ ∈ [5, ls.max()/3]
    for bi in range(MIN_SIDE, len(ls) - MIN_SIDE):
        tau_star = ls[bi]
        if tau_star < 5 or tau_star > ls.max() / 3: continue
        g1, a1 = np.polyfit(logL[:bi+1], logY[:bi+1], 1)
        g2, a2 = np.polyfit(logL[bi:], logY[bi:], 1)
        gamma_sh, gamma_lg = -g1, -g2
        if gamma_sh < gamma_lg - 0.05: continue       # enforce short ≥ long
        pred = np.concatenate([a1 + g1*logL[:bi+1], a2 + g2*logL[bi+1:]])
        r2b = 1 - ((logY - pred)**2).sum() / sstot
        if r2b > best[0]:
            best = (r2b, tau_star, gamma_sh, gamma_lg, a1, a2, bi)
    r2_br, tau_star, g_short, g_long, a_short, a_long, bi = best
    has_break = tau_star is not None and (r2_br - r2_p) > 0.005

    summary.append((sid, rate, -b_e, r2_e, -g_p, r2_p, g_short or -g_p,
                    g_long or -g_p, tau_star or 0, r2_br))

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    tau_s = np.logspace(0, np.log10(ls.max()), 300)
    y_lo_s = max(ys.min() * 0.5, 1e-4)
    y_hi_s = min(ys.max() * 2.0, 1.5)
    clip_s = lambda yy: np.maximum(yy, y_lo_s)
    ax = axs[0]
    ax.semilogy(ls, clip_s(ys), 'o', ms=4, color="#1f77b4", label="data")
    ax.semilogy(tau_s, clip_s(np.exp(a_e + b_e*tau_s)), 'r-', lw=1.2,
                label=f"exp β={-b_e:.4f}  R²={r2_e:.2f}")
    ax.semilogy(tau_s, clip_s(0.263 * np.exp(-0.05*tau_s)), 'r:', lw=1.2,
                label="sim β=0.05")
    ax.set_xlabel('lag τ'); ax.set_ylabel('ACF(τ)'); ax.set_title('log-linear')
    ax.set_ylim(y_lo_s, y_hi_s); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    ax = axs[1]
    ax.loglog(ls, clip_s(ys), 'o', ms=4, color="#1f77b4", label="data")
    ax.loglog(tau_s, clip_s(np.exp(a_p + g_p*np.log(tau_s))), 'g-', lw=1.5,
              label=f"single power γ={-g_p:.2f}  R²={r2_p:.2f}")
    if has_break:
        tau_left  = np.logspace(0, np.log10(tau_star), 120)
        tau_right = np.logspace(np.log10(tau_star), np.log10(ls.max()), 120)
        ax.loglog(tau_left,  clip_s(np.exp(a_short + -g_short*np.log(tau_left))),
                  color='#ff7f0e', lw=2, label=f"broken: γ_short={g_short:.2f}")
        ax.loglog(tau_right, clip_s(np.exp(a_long  + -g_long*np.log(tau_right))),
                  color='#9467bd', lw=2, label=f"broken: γ_long={g_long:.2f}")
        ax.axvline(tau_star, color='k', lw=0.5, ls='--')
        ax.text(tau_star, y_hi_s * 0.6, f'τ*={tau_star}',
                rotation=90, fontsize=8, color='#444', va='top')
    ax.set_xlabel('lag τ'); ax.set_ylabel('ACF(τ)'); ax.set_title('log-log')
    ax.set_ylim(y_lo_s, y_hi_s); ax.legend(fontsize=8); ax.grid(alpha=0.25, which="both")
    fig.suptitle(f'ses{sid}: kernel ACF fit  (λ_bar={rate:.3f})', fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{KDIR}/ses{sid:02d}.png", dpi=100)
    plt.close(fig)

# Summary figure: broken power-law γ_short vs γ_long per session + R² gains
if summary:
    S = np.array(summary)
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axs[0].bar(S[:, 0] - 0.2, S[:, 6], width=0.4, color="#ff7f0e", label='γ_short (short lags)')
    axs[0].bar(S[:, 0] + 0.2, S[:, 7], width=0.4, color="#9467bd", label='γ_long (long lags)')
    axs[0].set_ylabel(r'power-law exponent $\gamma$')
    axs[0].set_title('Broken power-law per session — short vs long regime')
    axs[0].grid(alpha=0.25); axs[0].legend()
    axs[1].bar(S[:, 0], S[:, 8], color="#17becf")
    axs[1].set_ylabel(r'breakpoint $\tau^*$ (rows)')
    axs[1].set_yscale('log'); axs[1].grid(alpha=0.25, which='both')
    axs[2].bar(S[:, 0] - 0.3, S[:, 3], width=0.3, color="#d62728", label=r'R² exp')
    axs[2].bar(S[:, 0],       S[:, 5], width=0.3, color="#2ca02c", label=r'R² power')
    axs[2].bar(S[:, 0] + 0.3, S[:, 9], width=0.3, color="#9467bd", label=r'R² broken-power')
    axs[2].set_xlabel('session id'); axs[2].set_ylabel('R² (log-space fit)')
    axs[2].set_ylim(0, 1.05); axs[2].legend(); axs[2].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/22_kernel_summary_all_sessions.png", dpi=120)
    plt.close(fig)
    print(f"\n{'ses':>4} {'rate':>7} {'exp_β':>10} {'R²_exp':>8} {'pow_γ':>7} {'R²_pow':>8} "
          f"{'γ_sh':>6} {'γ_lng':>6} {'τ*':>5} {'R²_br':>7}")
    for row in summary:
        print(f"  {int(row[0]):>3} {row[1]:>7.4f} {row[2]:>10.5f} {row[3]:>8.3f} "
              f"{row[4]:>7.3f} {row[5]:>8.3f} {row[6]:>6.2f} {row[7]:>6.2f} "
              f"{int(row[8]):>5d} {row[9]:>7.3f}")
    p_win = (S[:, 5] > S[:, 3]).mean() * 100
    print(f"\nPower-law beats exp in {(S[:,5]>S[:,3]).sum()}/{len(S)} sessions ({p_win:.0f}%)")
    print(f"Mean γ (single)  = {S[:, 4].mean():.3f} ± {S[:, 4].std():.3f}")
    print(f"Mean γ_short     = {S[:, 6].mean():.3f} ± {S[:, 6].std():.3f}")
    print(f"Mean γ_long      = {S[:, 7].mean():.3f} ± {S[:, 7].std():.3f}")
    print(f"Mean R² (power)  = {S[:, 5].mean():.3f}")
    print(f"Mean R² (broken) = {S[:, 9].mean():.3f}")
    print(f"Broken-power beats single-power in {(S[:,9]>S[:,5]+0.01).sum()}/{len(S)} sessions")

print(f"\nwrote figs:  19_kernel_acf.png, 20_kernel_per_session.png,")
print(f"             21_kernel_residual.png, 22_kernel_summary_all_sessions.png,")
print(f"             kernel/ses00..ses61.png")
