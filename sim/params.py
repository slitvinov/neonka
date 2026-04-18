"""Load Hawkes params per session and plot as function of session id."""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = f"{HERE}/figs"
os.makedirs(FIGS, exist_ok=True)

D = 8    # Strict 8-D fit: visible event types only. HP events skipped —
         # they are observation artifacts of the truncation to 8 visible levels.
TYPES = ['tp_a','tp_b','tm_a','tm_b','dp_a','dp_b','dm_a','dm_b']
PAIRS = [('tp', 0, 1), ('tm', 2, 3), ('dp', 4, 5), ('dm', 6, 7)]

def load_params(s):
    p = f'/tmp/hawkes{s}.params'
    if not os.path.exists(p): return None
    beta = 0.05; mu = np.zeros(D); alpha = np.zeros((D, D))
    with open(p) as f:
        for ln in f:
            parts = ln.split()
            if not parts: continue
            if parts[0] == 'beta': beta = float(parts[1])
            elif parts[0] == 'mu':
                c = int(parts[1])
                if c < D: mu[c] = float(parts[2])
            elif parts[0] == 'alpha':
                c = int(parts[1]); j = int(parts[2])
                if c < D and j < D: alpha[c, j] = float(parts[3])
    return beta, mu, alpha

B = np.zeros(62)
MU = np.zeros((62, D))
A = np.zeros((62, D, D))
for s in range(62):
    r = load_params(s)
    if r is None: continue
    B[s], MU[s], A[s] = r

# Derived
row_sum_over_beta = A.sum(axis=2) / B[:, None]          # α row sums / β per type
branch_rho = np.array([np.max(np.abs(np.linalg.eigvals(A[s]/B[s])))
                       if B[s] > 0 else 0 for s in range(62)])
stat_lam = np.zeros((62, D))
for s in range(62):
    I = np.eye(D)
    try:
        stat_lam[s] = np.linalg.solve(I - A[s]/B[s], MU[s])
    except np.linalg.LinAlgError:
        pass

# --- 1) β per session ---
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.bar(np.arange(62), B, color="#1f77b4")
ax.set_xlabel("session id")
ax.set_ylabel(r"Hawkes decay $\beta$")
ax.set_title(r"Fitted $\beta$ per session (half-life $\approx \ln 2/\beta$)")
ax.axhline(B.mean(), color='k', lw=1, ls='--', label=f"mean = {B.mean():.4f}")
ax.legend(); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(f"{FIGS}/11_beta_per_sess.png", dpi=120); plt.close(fig)

# --- 2) μ per type per session — pool ask+bid (symmetric) ---
fig, axs = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
for ax, (name, ia, ib) in zip(axs.flat, PAIRS):
    pooled = MU[:, ia] + MU[:, ib]
    ax.bar(np.arange(62), pooled, color="#2ca02c")
    ax.set_title(f"μ[{name}_a] + μ[{name}_b]"); ax.grid(alpha=0.25)
    ax.set_ylabel(r"$\mu_a + \mu_b$")
for ax in axs[-1]: ax.set_xlabel("session id")
fig.suptitle(r"Baseline rate $\mu_c$ per event pair (ask+bid pooled)")
fig.tight_layout(); fig.savefig(f"{FIGS}/12_mu_per_type.png", dpi=120); plt.close(fig)

# --- 3) μ total + ratios (ask vs bid) ---
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axs[0].bar(np.arange(62), MU.sum(axis=1), color="#9467bd")
axs[0].set_ylabel(r"$\sum_c \mu_c$ (baseline total rate)")
axs[0].set_title("μ aggregate per session")
axs[0].grid(alpha=0.25)

# Pair ratios (ask/bid)
COLOURS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
for (name, ia, ib), col in zip(PAIRS, COLOURS):
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(MU[:, ib] > 0, MU[:, ia] / np.maximum(MU[:, ib], 1e-12), np.nan)
    axs[1].plot(np.arange(62), r, 'o-', label=f"μ({name}_a)/μ({name}_b)", color=col)
axs[1].axhline(1, color='k', lw=1)
axs[1].set_xlabel("session id")
axs[1].set_ylabel("ask / bid ratio")
axs[1].set_title("Bid/ask symmetry of μ per session")
axs[1].legend(fontsize=9); axs[1].grid(alpha=0.25)
fig.tight_layout(); fig.savefig(f"{FIGS}/13_mu_aggregate_and_sym.png", dpi=120); plt.close(fig)

# --- 4) Branching ratio ρ(α/β) ---
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.bar(np.arange(62), branch_rho, color="#d62728")
ax.axhline(1, color='k', lw=1)
ax.axhline(branch_rho.mean(), color='k', lw=1, ls='--', label=f"mean = {branch_rho.mean():.3f}")
ax.set_xlabel("session id"); ax.set_ylabel(r"$\rho(\alpha/\beta)$")
ax.set_title("Branching ratio per session (must be < 1 for stationarity)")
ax.set_ylim(0, 1.05)
ax.legend(); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(f"{FIGS}/14_branching_ratio.png", dpi=120); plt.close(fig)

# --- 5) α row sums per type — pool ask+bid (average of pair) ---
fig, axs = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
for ax, (name, ia, ib) in zip(axs.flat, PAIRS):
    pooled = 0.5 * (row_sum_over_beta[:, ia] + row_sum_over_beta[:, ib])
    ax.bar(np.arange(62), pooled, color="#ff7f0e")
    ax.set_title(f"½(Σ_j α[{name}_a, j] + Σ_j α[{name}_b, j]) / β")
    ax.axhline(1, color='k', lw=0.8)
    ax.grid(alpha=0.25); ax.set_ylim(0, 1.1)
for ax in axs[-1]: ax.set_xlabel("session id")
fig.suptitle(r"Row-sum of $\alpha/\beta$ averaged over ask/bid pair (closer to 1 = more self/cross-excited)")
fig.tight_layout(); fig.savefig(f"{FIGS}/15_alpha_rowsums.png", dpi=120); plt.close(fig)

# --- 6) Stationary λ per type — pool ask+bid ---
fig, axs = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
for ax, (name, ia, ib) in zip(axs.flat, PAIRS):
    pooled = stat_lam[:, ia] + stat_lam[:, ib]
    ax.bar(np.arange(62), pooled, color="#17becf")
    ax.set_title(f"λ_stat[{name}_a] + λ_stat[{name}_b]"); ax.grid(alpha=0.25)
    ax.set_ylabel(r"$\lambda^{stat}_a + \lambda^{stat}_b$")
for ax in axs[-1]: ax.set_xlabel("session id")
fig.suptitle(r"Stationary event rate $\lambda_c^{stat}$ per pair (ask+bid pooled)")
fig.tight_layout(); fig.savefig(f"{FIGS}/16_lambda_stat.png", dpi=120); plt.close(fig)

# --- 7) α heatmap averaged across sessions ---
A_mean = A.mean(axis=0) / B.mean()
A_std_over_mean = A.std(axis=0) / (A.mean(axis=0) + 1e-9)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
im0 = axs[0].imshow(A_mean, cmap="viridis", aspect="auto")
axs[0].set_title(r"mean $\alpha_{ij}/\beta$ across 62 sessions")
axs[0].set_xticks(range(D)); axs[0].set_xticklabels(TYPES, rotation=45, ha='right')
axs[0].set_yticks(range(D)); axs[0].set_yticklabels(TYPES)
axs[0].set_xlabel("source j"); axs[0].set_ylabel("target i")
plt.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(A_std_over_mean, cmap="magma", aspect="auto", vmax=1.5)
axs[1].set_title(r"CV of $\alpha_{ij}$ across sessions (std/mean)")
axs[1].set_xticks(range(D)); axs[1].set_xticklabels(TYPES, rotation=45, ha='right')
axs[1].set_yticks(range(D)); axs[1].set_yticklabels(TYPES)
axs[1].set_xlabel("source j")
plt.colorbar(im1, ax=axs[1])
fig.tight_layout(); fig.savefig(f"{FIGS}/17_alpha_heatmap.png", dpi=120); plt.close(fig)

# --- 8) summary stats printout ---
print(f"--- Summary across 62 sessions ---")
print(f"β        : mean={B.mean():.5f}  std={B.std():.5f}  range=[{B.min():.5f}, {B.max():.5f}]")
print(f"Σμ       : mean={MU.sum(axis=1).mean():.4f}  std={MU.sum(axis=1).std():.4f}")
print(f"ρ(α/β)   : mean={branch_rho.mean():.4f}  std={branch_rho.std():.4f}  range=[{branch_rho.min():.3f}, {branch_rho.max():.3f}]")
print(f"Σλ_stat  : mean={stat_lam.sum(axis=1).mean():.4f}  std={stat_lam.sum(axis=1).std():.4f}")
print()
print(f"Bid/ask symmetry of μ (ask/bid ratio):")
for pair, i, j in PAIRS:
    mask = MU[:, j] > 0
    r = MU[mask, i] / MU[mask, j]
    print(f"  μ(.{pair}_a)/μ(.{pair}_b): mean={r.mean():.4f}  std={r.std():.4f}  (N={mask.sum()})")
print()
print(f"7 figures written to {FIGS}/11_..17_")
