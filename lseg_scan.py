"""Scan LSEG.L across all history for best time offset matching our 63-session median prices.
Our median prices are in real units (real_price = aR[0] / 4). We also allow a constant price shift.
"""
import sys, os, subprocess
import numpy as np

os.makedirs("/tmp/lsegscan", exist_ok=True)

our_meds = []
for ses in range(63):
    d = f"data/tables/{ses:02d}"
    cmd = f"./session -D data/train.raw -S data/sessions.raw -s {ses} | ./pairs 2>/dev/null | head -c 196000"
    p = f"/tmp/lsegscan/_med_{ses}.txt"
    if not os.path.exists(p):
        r = subprocess.run(
            f"./session -D data/train.raw -S data/sessions.raw -s {ses} | "
            f"python3 -c 'import sys,numpy as np; "
            f"a=np.frombuffer(sys.stdin.buffer.read(),dtype=np.int32).reshape(-1,49); "
            f"mid=(a[:,0]+a[:,8])/8.0; print(np.median(mid))' > {p}",
            shell=True, check=False)
    our_meds.append(float(open(p).read().strip()))

our_meds = np.array(our_meds)
print(f"our medians n={len(our_meds)}, range [{our_meds.min():.1f}, {our_meds.max():.1f}], median {np.median(our_meds):.1f}")

try:
    import yfinance as yf
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "--break-system-packages", "--user", "yfinance", "-q"])
    import yfinance as yf

df = yf.download("LSEG.L", start="2019-01-01", end="2026-04-17", progress=False, auto_adjust=False)
print(f"LSEG.L: {len(df)} days, {df.index[0].date()} to {df.index[-1].date()}")
close = df['Close'].to_numpy().flatten()
print(f"close range [{close.min():.0f}, {close.max():.0f}]")

best = None
N = len(our_meds)
for off in range(0, len(close) - N):
    window = close[off:off+N]
    shift = np.median(our_meds) - np.median(window)
    rmse = float(np.sqrt(np.mean((our_meds - (window + shift))**2)))
    corr = float(np.corrcoef(our_meds, window)[0, 1]) if np.std(window) > 0 else -1
    if best is None or (rmse < best[0] and corr > 0.3):
        best = (rmse, corr, off, shift, window.copy(), df.index[off].date(), df.index[off+N-1].date())

if best:
    print(f"\nbest: RMSE={best[0]:.1f} corr={best[1]:+.3f}")
    print(f"  window: {best[5]} to {best[6]}")
    print(f"  price shift: {best[3]:+.1f} pence (we add to LSEG)")
    print(f"  LSEG range: [{best[4].min():.0f}, {best[4].max():.0f}]")
    print(f"  our range:  [{our_meds.min():.0f}, {our_meds.max():.0f}]")
    print(f"\nfirst 10 sessions:")
    print(f"  {'ses':>3} {'LSEG':>8} {'+shift':>8} {'ours':>8} {'diff':>8}")
    for i in range(min(15, N)):
        lseg = best[4][i]
        shifted = lseg + best[3]
        print(f"  {i:>3} {lseg:>8.1f} {shifted:>8.1f} {our_meds[i]:>8.1f} {our_meds[i]-shifted:>+8.1f}")
