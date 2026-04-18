"""Correlate per-session R² with regime features (dp_frac, tp_frac, rvol, etc.)
If crisis sessions drive our R², we know where to focus next.
Also checks if within-session regime drift can be detected.
"""
import numpy as np, os, subprocess, sys

raw = np.fromfile("data/train.raw", dtype=np.int32).reshape(-1, 49)
sess = np.fromfile("data/sessions.raw", dtype=np.int64)
aR0, bR0 = raw[:,0], raw[:,8]
aS0, bS0 = raw[:,16], raw[:,24]
aN0, bN0 = raw[:,32], raw[:,40]
mid = (aR0+bR0)/2.0; sp = aR0-bR0

def load_rates(ses):
    d = f"data/tables/{ses:02d}"
    r = {}
    for name in ["tp","tm","dp","dm"]:
        p = f"{d}/{name}.rates"
        if os.path.exists(p):
            arr = np.loadtxt(p)
            if arr.ndim == 1: arr = arr.reshape(1, -1)
            r[name] = arr[:, 1].mean()
    return r

per_r2_str = """\
ses=61 5.42
ses=57 4.49
ses=62 4.48
ses=58 3.69
ses=55 3.14
ses=56 2.97
ses=60 2.87
ses=46 2.84
ses=50 2.84
ses=44 2.72
ses=54 2.61
ses=59 2.60
ses=49 2.37
ses=39 2.31
ses=53 2.28
ses=52 2.18
ses=45 2.02
ses=40 1.97
ses=16 1.94
ses=47 1.84
"""
r2_map = {}
for line in per_r2_str.strip().split("\n"):
    p = line.split()
    r2_map[int(p[0].split("=")[1])] = float(p[1])

print(f"{'ses':>3} {'R2%':>5} {'regime':<7} {'dp_frac':>7} {'tp_frac':>7} {'rvol':>6} {'medsp':>5} {'len':>7} {'|ret%|':>6}")
for s in sorted(r2_map, key=lambda k: -r2_map[k]):
    r = load_rates(s)
    if not r: continue
    tp, tm, dp, dm = r['tp'], r['tm'], r['dp'], r['dm']
    tp_frac = tp/(tp+tm); dp_frac = dp/(dp+dm)
    a, b = sess[s], sess[s+1]
    m = mid[a:b]
    rvol = np.std(np.diff(m))*np.sqrt(1000)
    msp = int(np.median(sp[a:b]))
    ret_pct = abs((m[-1]/m[0]-1)*100)
    regime = "pre" if s<=43 else ("calm-end" if s<=51 else ("crisis" if s<=59 else "post"))
    print(f"{s:>3} {r2_map[s]:>5.2f} {regime:<7} {dp_frac:>7.3f} {tp_frac:>7.3f} "
          f"{rvol:>6.1f} {msp:>5} {b-a:>7,} {ret_pct:>6.2f}")

regimes = {"pre": list(range(0,44)), "calm-end": list(range(44,52)), "crisis": list(range(52,60)), "post": list(range(60,63))}
print("\n=== Regime averages ===")
for name, sl in regimes.items():
    r2s = [r2_map[s] for s in sl if s in r2_map]
    print(f"  {name:<10} n_in_top20={len(r2s)}/{len(sl)}  mean R2 in top-20={np.mean(r2s) if r2s else 0:.2f}")

print("\n=== Within-session regime drift — split each session into halves ===")
print(f"{'ses':>3} {'1st_rvol':>8} {'2nd_rvol':>8} {'1st_sp':>6} {'2nd_sp':>6} {'1st_mvrate':>10} {'2nd_mvrate':>10}")
for s in [47, 48, 53, 54, 56, 61, 62]:
    a, b = sess[s], sess[s+1]
    half = (a+b)//2
    m1, m2 = mid[a:half], mid[half:b]
    sp1, sp2 = sp[a:half], sp[half:b]
    rv1 = np.std(np.diff(m1))*np.sqrt(1000)
    rv2 = np.std(np.diff(m2))*np.sqrt(1000)
    mv1 = (np.diff(m1)!=0).mean()*1000
    mv2 = (np.diff(m2)!=0).mean()*1000
    print(f"{s:>3} {rv1:>8.1f} {rv2:>8.1f} {np.median(sp1):>6.0f} {np.median(sp2):>6.0f} {mv1:>10.1f} {mv2:>10.1f}")
