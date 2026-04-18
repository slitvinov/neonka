"""Generate all synthetic-data-evidence plots via matplotlib.
Usage: python3 sim/plot.py (assumes sim/evidence.py has been run first).
"""
import os, sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = f"{HERE}/data"
FIGS = f"{HERE}/figs"
os.makedirs(FIGS, exist_ok=True)

def load(path, cols=None):
    return np.loadtxt(path, comments="#", dtype=float)

# -------- 1) Pooled Δmid distribution (log y) --------
d = load(f"{DATA}/dmid_pooled.dat")
k, P = d[:, 0].astype(int), d[:, 2]
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(k, P, color="#1f77b4", alpha=0.8)
ax.set_yscale("log")
ax.set_xlabel("Δmid (ticks per IDLE row)")
ax.set_ylabel("P(Δmid = k)")
ax.set_title("Pooled one-step Δmid distribution (log y) — 2M+ observations, 62 sessions")
ax.axvline(0, color="k", lw=0.5)
ax.set_xticks(range(-10, 11, 2))
ax.grid(True, which="both", alpha=0.25)
fig.tight_layout()
fig.savefig(f"{FIGS}/01_dmid_pooled.png", dpi=120)
plt.close(fig)

# -------- 2) P(+k) vs P(-k) mirror overlay --------
mask_neg = d[:, 0] < 0
mask_pos = d[:, 0] > 0
ks_n = -d[mask_neg, 0].astype(int)[::-1]
Ps_n = d[mask_neg, 2][::-1]
ks_p = d[mask_pos, 0].astype(int)
Ps_p = d[mask_pos, 2]
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(ks_n, Ps_n, s=80, marker="o", c="#d62728", label="P(−k)", zorder=3)
ax.scatter(ks_p, Ps_p, s=80, marker="^", c="#2ca02c", label="P(+k)", zorder=3)
ax.set_yscale("log")
ax.set_xlabel("|k|")
ax.set_ylabel("P(|Δmid| = k)")
ax.set_title("P(+k) vs P(−k) overlaid; indistinguishable in log scale")
ax.legend()
ax.grid(True, which="both", alpha=0.25)
fig.tight_layout()
fig.savefig(f"{FIGS}/02_dmid_mirror.png", dpi=120)
plt.close(fig)

# -------- 3) Per-session log2(P+1/P-1) --------
d = load(f"{DATA}/dmid_per_sess.dat")
s, r1 = d[:, 0].astype(int), d[:, 1]
N = d[:, 4]
theo = 2 * np.sqrt(0.088 * 0.912 / np.median(N)) / (0.088 * np.log(2))
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(s, r1, color="#1f77b4", alpha=0.85)
ax.axhline(0, color="k", lw=1)
ax.axhline(+theo, color="#888888", lw=1, ls="--")
ax.axhline(-theo, color="#888888", lw=1, ls="--")
ax.text(61, theo + 0.005, f"±1σ (50/50 Bernoulli, N≈{int(np.median(N))})",
        ha="right", va="bottom", color="#555555", fontsize=9)
ax.set_xlabel("session id")
ax.set_ylabel(r"$\log_2(P(+1)/P(-1))$")
ax.set_title("Per-session up/down asymmetry — zero-centered, no regime trend")
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(f"{FIGS}/03_dmid_per_session_ratio.png", dpi=120)
plt.close(fig)

# -------- 4) Pooled log2(P+k/P-k) at each |k| --------
d = load(f"{DATA}/sym_pooled_ratio.dat")
ks, r, z = d[:, 0].astype(int), d[:, 3], d[:, 4]
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(ks, r, color="#2ca02c", alpha=0.85)
ax.axhline(0, color="k", lw=1)
for xi, yi, zi in zip(ks, r, z):
    ax.text(xi, yi + 0.015 * np.sign(yi if yi else 1),
            f"z={zi:+.1f}", ha="center", va="bottom" if yi >= 0 else "top",
            fontsize=9, color="#333333")
ax.set_xlabel("|k|")
ax.set_ylabel(r"$\log_2(N_+ / N_-)$ pooled")
ax.set_title("Pooled up/down ratio at each |k| — all within noise (z < 2.5)")
ax.set_xticks(ks)
ax.set_ylim(-0.35, 0.35)
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(f"{FIGS}/04_sym_pooled_ratio.png", dpi=120)
plt.close(fig)

# -------- 5) Event balance bid↔ask per session --------
d = load(f"{DATA}/event_balance.dat")
ss = d[:, 0].astype(int)
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(ss, d[:, 1], s=25, c="#d62728", label="tp_a/tp_b")
ax.scatter(ss, d[:, 2], s=25, c="#1f77b4", label="tm_a/tm_b")
ax.scatter(ss, d[:, 3], s=25, c="#2ca02c", label="dp_a/dp_b")
ax.scatter(ss, d[:, 4], s=25, c="#ff7f0e", label="dm_a/dm_b")
ax.axhline(0, color="k", lw=1)
ax.set_xlabel("session id")
ax.set_ylabel(r"$\log_2(N_a / N_b)$")
ax.set_title("Bid↔ask event-count symmetry per session")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.25)
ax.set_ylim(-0.25, 0.25)
fig.tight_layout()
fig.savefig(f"{FIGS}/05_event_balance.png", dpi=120)
plt.close(fig)

# -------- 6) Fingerprints (vol, spread, ev_rate) --------
d = load(f"{DATA}/fingerprints.dat")
s = d[:, 0].astype(int)
fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
axs[0].bar(s, d[:, 6], color="#1f77b4"); axs[0].set_ylabel("vol(Δmid)")
axs[0].set_title("Per-session fingerprints — regime break at ses52")
axs[1].bar(s, d[:, 7], color="#d62728"); axs[1].set_ylabel("median spread")
axs[2].bar(s, d[:, 2], color="#2ca02c"); axs[2].set_ylabel("events / idle")
axs[2].set_xlabel("session id")
for ax in axs: ax.grid(True, alpha=0.25)
axs[2].set_ylim(0.82, 0.98)
fig.tight_layout()
fig.savefig(f"{FIGS}/06_fingerprints.png", dpi=120)
plt.close(fig)

# -------- 7) mid return per session --------
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(s, d[:, 5], color="#9467bd")
ax.axhline(0, color="k", lw=1)
ax.set_xlabel("session id")
ax.set_ylabel("mid_end − mid_start (ticks)")
ax.set_title("Per-session mid return — aggregate drift across 62 sessions ≈ 0")
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(f"{FIGS}/07_mid_return.png", dpi=120)
plt.close(fig)

# -------- 8) SMOKING GUN: cumulative a-side vs b-side in sample sessions --------
SAMPLES = [0, 30, 45, 56]
fig, axs = plt.subplots(2, 2, figsize=(12, 7.5), sharex=False)
for ax, sid in zip(axs.flat, SAMPLES):
    try:
        d = np.loadtxt(f"{DATA}/cum_ses{sid}.dat", comments="#")
    except OSError:
        ax.text(0.5, 0.5, f"no data for ses{sid}", ha="center", va="center",
                transform=ax.transAxes); continue
    t = d[:, 0]
    tp_a, tp_b = d[:, 1], d[:, 2]
    tm_a, tm_b = d[:, 3], d[:, 4]
    ax.plot(t, tp_a, lw=1.5, color="#d62728", label="tp_a")
    ax.plot(t, tp_b, lw=1.5, ls="--", color="#ff7f0e", label="tp_b")
    ax.plot(t, tm_a, lw=1.5, color="#1f77b4", label="tm_a")
    ax.plot(t, tm_b, lw=1.5, ls="--", color="#17becf", label="tm_b")
    ax.set_title(f"ses{sid}")
    ax.set_xlabel("row (time)")
    ax.set_ylabel("cumulative event count")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25)
fig.suptitle("Cumulative tp/tm for ask vs bid per session — curves track tightly", fontsize=12)
fig.tight_layout()
fig.savefig(f"{FIGS}/08_smoking_gun_cumcount.png", dpi=120)
plt.close(fig)

# -------- 9) Cumulative a−b difference (the "overlap quality") --------
fig, axs = plt.subplots(2, 2, figsize=(12, 7.5), sharex=False)
for ax, sid in zip(axs.flat, SAMPLES):
    try:
        d = np.loadtxt(f"{DATA}/cum_ses{sid}.dat", comments="#")
    except OSError:
        continue
    t = d[:, 0]
    ax.plot(t, d[:,  9], lw=1.5, color="#d62728", label="tp_a − tp_b")
    ax.plot(t, d[:, 10], lw=1.5, color="#1f77b4", label="tm_a − tm_b")
    ax.plot(t, d[:, 11], lw=1.5, color="#2ca02c", label="dp_a − dp_b")
    ax.plot(t, d[:, 12], lw=1.5, color="#ff7f0e", label="dm_a − dm_b")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"ses{sid}")
    ax.set_xlabel("row (time)")
    ax.set_ylabel("N_cum(ask) − N_cum(bid)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)
fig.suptitle("Bid↔ask cumulative-count DIFFERENCE stays bounded — smoking gun of symmetric generator",
             fontsize=12)
fig.tight_layout()
fig.savefig(f"{FIGS}/09_smoking_gun_diff.png", dpi=120)
plt.close(fig)

# -------- 10) max|diff| across all 62 sessions --------
d = load(f"{DATA}/cum_gap_summary.dat")
s = d[:, 0].astype(int)
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axs[0].plot(s, d[:, 1], 'o-', label="max|tp_diff|", color="#d62728")
axs[0].plot(s, d[:, 2], 'o-', label="max|tm_diff|", color="#1f77b4")
axs[0].plot(s, d[:, 3], 'o-', label="max|dp_diff|", color="#2ca02c")
axs[0].plot(s, d[:, 4], 'o-', label="max|dm_diff|", color="#ff7f0e")
axs[0].set_ylabel("max |cum(a) − cum(b)|")
axs[0].set_title("Cumulative a−b gap vs session; end-of-session residual (bottom) is tiny")
axs[0].legend(fontsize=9); axs[0].grid(alpha=0.25)
axs[1].plot(s, d[:, 5], 'o-', label="end tp_diff", color="#d62728")
axs[1].plot(s, d[:, 6], 'o-', label="end tm_diff", color="#1f77b4")
axs[1].plot(s, d[:, 7], 'o-', label="end dp_diff", color="#2ca02c")
axs[1].plot(s, d[:, 8], 'o-', label="end dm_diff", color="#ff7f0e")
axs[1].axhline(0, color='k', lw=0.8)
axs[1].set_xlabel("session id")
axs[1].set_ylabel("end-of-session a − b")
axs[1].legend(fontsize=9); axs[1].grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{FIGS}/10_cum_gap_summary.png", dpi=120)
plt.close(fig)

print(f"wrote 10 figures → {FIGS}/")
