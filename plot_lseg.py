import subprocess, sys, numpy as np
try: import yfinance as yf
except ImportError:
    subprocess.run([sys.executable,"-m","pip","install","--break-system-packages","--user","yfinance","-q"])
    import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

our = np.array([float(open(f"/tmp/lsegscan/_med_{s}.txt").read()) for s in range(63)])

df = yf.download("LSEG.L", start="2019-01-01", end="2026-04-17", progress=False, auto_adjust=False)
close = df['Close'].to_numpy().flatten()
dates = df.index

N = 63
best_rmse = 1e18
best_shift_info = None
for off in range(0, len(close)-N):
    w = close[off:off+N]
    rmse_noshift = float(np.sqrt(np.mean((our-w)**2)))
    shift = float(np.median(our) - np.median(w))
    rmse_shift = float(np.sqrt(np.mean((our-(w+shift))**2)))
    corr = float(np.corrcoef(our,w)[0,1]) if np.std(w)>0 else -1
    if rmse_noshift < best_rmse:
        best_rmse = rmse_noshift
        best_noshift = (off, w.copy(), dates[off], dates[off+N-1], corr)

off_a = best_noshift[0]
w_a = best_noshift[1]

best_shift_rmse = 1e18
for off in range(0, len(close)-N):
    w = close[off:off+N]
    shift = float(np.median(our) - np.median(w))
    rmse_shift = float(np.sqrt(np.mean((our-(w+shift))**2)))
    corr = float(np.corrcoef(our,w)[0,1]) if np.std(w)>0 else -1
    if rmse_shift < best_shift_rmse and corr > 0.3:
        best_shift_rmse = rmse_shift
        best_shift = (off, w.copy(), shift, dates[off], dates[off+N-1], corr)

off_b, w_b, sh_b, d0_b, d1_b, cb = best_shift

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

ax = axes[0]
ax.plot(range(63), our, 'o-', label='Our data (session medians)', color='C0', linewidth=2)
ax.plot(range(63), w_a, 's-', label=f'LSEG.L {best_noshift[2].date()}→{best_noshift[3].date()} (no shift)',
        color='C1', alpha=0.8)
ax.set_xlabel('Session index (0..62)')
ax.set_ylabel('Price (pence)')
ax.set_title(f'Best match no-shift: LSEG.L  RMSE={best_rmse:.0f}  corr={best_noshift[4]:+.3f}')
ax.grid(alpha=0.3); ax.legend()

ax = axes[1]
ax.plot(range(63), our, 'o-', label='Our data', color='C0', linewidth=2)
ax.plot(range(63), w_b + sh_b, 's-',
        label=f'LSEG.L {d0_b.date()}→{d1_b.date()}  shift={sh_b:+.0f}p', color='C2', alpha=0.8)
ax.plot(range(63), w_b, 'x--', label=f'LSEG.L raw (no shift, 2021)', color='C3', alpha=0.5)
ax.set_xlabel('Session index (0..62)')
ax.set_ylabel('Price (pence)')
ax.set_title(f'Best match with shift: LSEG.L 2021  RMSE={best_shift_rmse:.0f}  corr={cb:+.3f}')
ax.grid(alpha=0.3); ax.legend()

plt.tight_layout()
plt.savefig("/Users/lisergey/neonka/lseg_compare.png", dpi=110)
print(f"no-shift: LSEG {best_noshift[2].date()}→{best_noshift[3].date()} RMSE={best_rmse:.0f} corr={best_noshift[4]:+.3f}")
print(f"w/shift:  LSEG {d0_b.date()}→{d1_b.date()} shift={sh_b:+.0f}p RMSE={best_shift_rmse:.0f} corr={cb:+.3f}")
print(f"saved: lseg_compare.png")
