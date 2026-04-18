import subprocess, sys, numpy as np
try: import yfinance as yf
except ImportError:
    subprocess.run([sys.executable,"-m","pip","install","--break-system-packages","--user","yfinance","-q"])
    import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

raw = np.fromfile("data/train.raw", dtype=np.int32).reshape(-1, 49)
sess = np.fromfile("data/sessions.raw", dtype=np.int64)
n = len(raw)
print(f"raw rows {n}, sess boundaries {len(sess)}")

mid_real = (raw[:,0] + raw[:,8]) / 8.0

STRIDE = 500
idx = np.arange(0, n, STRIDE)
mid_thin = mid_real[idx]

our_meds = np.array([float(open(f"/tmp/lsegscan/_med_{s}.txt").read()) for s in range(63)])

df = yf.download("LSEG.L", start="2024-04-01", end="2024-07-20", progress=False, auto_adjust=False)
lseg_24 = df['Close'].to_numpy().flatten()[:63]
df2 = yf.download("LSEG.L", start="2021-07-01", end="2021-10-15", progress=False, auto_adjust=False)
lseg_21 = df2['Close'].to_numpy().flatten()[:63]
shift_21 = float(np.median(our_meds) - np.median(lseg_21))

sess_centers_tick = [(sess[s] + (sess[s+1] if s+1<len(sess) else n)) / 2 for s in range(63)]
sess_edges = list(sess) + [n]

fig, axes = plt.subplots(2, 1, figsize=(24, 12), sharex=True)

ax = axes[0]
ax.plot(idx, mid_thin, '-', color='C0', linewidth=0.4, alpha=0.7, label='Intraday mid (real units, stride=500)')
for b in sess_edges[1:-1]:
    ax.axvline(b, color='gray', alpha=0.25, linewidth=0.5)
ax.plot(sess_centers_tick, lseg_24, 's-', color='C1', markersize=4, linewidth=1.2,
        label='LSEG.L 2024-04-16→2024-07-15 close')
ax.plot(sess_centers_tick, our_meds, 'o', color='C2', markersize=3, label='Our session medians')
ax.set_ylabel('Price (pence)')
ax.set_title(f'63 sessions ({n:,} ticks). Vertical lines = session boundaries. LSEG.L 2024 overlay.')
ax.grid(alpha=0.3); ax.legend(loc='lower right', fontsize=9)

ax = axes[1]
ax.plot(idx, mid_thin, '-', color='C0', linewidth=0.4, alpha=0.7, label='Intraday mid')
for b in sess_edges[1:-1]:
    ax.axvline(b, color='gray', alpha=0.25, linewidth=0.5)
ax.plot(sess_centers_tick, lseg_21 + shift_21, 's-', color='C3', markersize=4, linewidth=1.2,
        label=f'LSEG.L 2021-07-13→2021-10-08 close + {shift_21:+.0f}p')
ax.plot(sess_centers_tick, our_meds, 'o', color='C2', markersize=3, label='Our session medians')
ax.set_xlabel('Tick index')
ax.set_ylabel('Price (pence)')
ax.set_title(f'Same intraday mid. LSEG.L 2021 (shifted +{shift_21:.0f}p) overlay.')
ax.grid(alpha=0.3); ax.legend(loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig("/Users/lisergey/neonka/lseg_sessions.png", dpi=120)
print(f"saved: lseg_sessions.png  (63 vertical lines at session boundaries)")
