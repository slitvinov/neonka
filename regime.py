"""Regime analysis focus — session 52-59 high-vol cluster.
Per-session summary: cumulative return, realized vol, event mix, spread, book thickness.
"""
import numpy as np
raw = np.fromfile("data/train.raw", dtype=np.int32).reshape(-1, 49)
sess = np.fromfile("data/sessions.raw", dtype=np.int64)
n = len(raw)

aR0, bR0 = raw[:,0], raw[:,8]
aS0, bS0 = raw[:,16], raw[:,24]
aN0, bN0 = raw[:,32], raw[:,40]
mid = (aR0+bR0)/2.0; sp = aR0-bR0

print(f"{'ses':>3} {'len':>8} {'mid_s':>7} {'mid_e':>7} {'ret%':>6} {'rvol':>6} "
      f"{'medsp':>5} {'sz1%':>5} {'N0':>4} {'Nd':>4} {'move/1k':>7}")
prev_end = None
for s in range(63):
    a, b = sess[s], sess[s+1]
    m = mid[a:b]
    if len(m) < 2: continue
    ret_pct = (m[-1]/m[0]-1)*100
    dm = np.diff(m)
    rvol = np.std(dm) * np.sqrt(1000)  # per-1k-tick vol
    msp = int(np.median(sp[a:b]))
    s1 = (aS0[a:b]==1).mean()*100
    mn0 = int(np.median(aN0[a:b]))
    mnd = int(np.median(raw[a:b, 33]))  # aN[1]
    rate = (dm!=0).mean()*1000
    gap = f"{m[0]-prev_end:+6.0f}" if prev_end is not None else "      "
    print(f"{s:>3} {b-a:>8,} {m[0]:>7.0f} {m[-1]:>7.0f} {ret_pct:>+6.2f} "
          f"{rvol:>6.1f} {msp:>5} {s1:>5.1f} {mn0:>4} {mnd:>4} {rate:>7.1f} gap={gap}")
    prev_end = m[-1]

print("\n=== total cumulative return ===")
all_mid = np.concatenate([mid[sess[s]:sess[s+1]] for s in range(63)])
first_per_sess = [mid[sess[s]] for s in range(63)]
last_per_sess  = [mid[sess[s+1]-1] for s in range(63)]
print(f"  overall return (last ses close / first ses open - 1): {(last_per_sess[-1]/first_per_sess[0]-1)*100:+.2f}%")
print(f"  overall return ignoring gaps (sum of intraday returns): "
      f"{sum((last_per_sess[s]/first_per_sess[s]-1) for s in range(63))*100:+.2f}%")
print(f"  overnight returns (gap component): "
      f"{sum((first_per_sess[s+1]/last_per_sess[s]-1) for s in range(62))*100:+.2f}%")

print("\n=== Peak and trough sessions by mid ===")
sess_min = [mid[sess[s]:sess[s+1]].min() for s in range(63)]
sess_max = [mid[sess[s]:sess[s+1]].max() for s in range(63)]
sess_med = [np.median(mid[sess[s]:sess[s+1]]) for s in range(63)]
imax = int(np.argmax(sess_max))
imin = int(np.argmin(sess_min))
print(f"  peak: session {imax}, mid_max={sess_max[imax]:.0f}")
print(f"  trough: session {imin}, mid_min={sess_min[imin]:.0f}")
print(f"  peak-to-trough: {(sess_min[imin]/sess_max[imax]-1)*100:+.2f}%")
