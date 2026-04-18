"""Event-mix shifts across regimes. Compare detailed-balance ratio, tp_frac, dp_frac.
Checks if crisis regime (52-59) has different market microstructure.
"""
import numpy as np, os, subprocess

REGIMES = {
    "pre   (0-43)":  list(range(0, 44)),
    "calm-end (44-51)": list(range(44, 52)),
    "crisis (52-59)":  list(range(52, 60)),
    "post  (60-62)":  list(range(60, 63)),
}

def load_rates(ses):
    d = f"data/tables/{ses:02d}"
    rates = {}
    for name in ["tp", "tm", "dp", "dm"]:
        p = f"{d}/{name}.rates"
        if os.path.exists(p):
            arr = np.loadtxt(p)
            if arr.ndim == 1: arr = arr.reshape(1, -1)
            # column 0 is sp, column 1 is rate. weighted by default: just take mean rate
            rates[name] = arr[:, 1].mean()
    return rates

rates_all = {}
for ses in range(63):
    r = load_rates(ses)
    if r: rates_all[ses] = r

print(f"{'regime':<20} {'tp':>6} {'tm':>6} {'dp':>6} {'dm':>6}  {'tp*dp':>7} {'tm*dm':>7} {'bal':>5}  {'tp_frac':>7} {'dp_frac':>7}")
for name, sess_list in REGIMES.items():
    avail = [s for s in sess_list if s in rates_all]
    if not avail: continue
    vals = {k: np.mean([rates_all[s][k] for s in avail]) for k in ["tp","tm","dp","dm"]}
    tp, tm, dp, dm = vals['tp'], vals['tm'], vals['dp'], vals['dm']
    balance = tp*dp / (tm*dm) if tm*dm > 0 else 0
    tp_frac = tp/(tp+tm) if tp+tm > 0 else 0
    dp_frac = dp/(dp+dm) if dp+dm > 0 else 0
    print(f"{name:<20} {tp:>6.3f} {tm:>6.3f} {dp:>6.3f} {dm:>6.3f}  {tp*dp:>7.3f} {tm*dm:>7.3f}  {balance:>5.2f}  {tp_frac:>7.3f} {dp_frac:>7.3f}")

print("\n=== Per-session rate trajectory (first 5, last 5, around ses 54) ===")
print(f"{'ses':>3} {'tp':>6} {'tm':>6} {'dp':>6} {'dm':>6} {'tp_frac':>7} {'dp_frac':>7} {'bal':>5}")
for s in list(range(5)) + list(range(42, 63)):
    if s not in rates_all: continue
    r = rates_all[s]
    tp, tm, dp, dm = r['tp'], r['tm'], r['dp'], r['dm']
    tp_frac = tp/(tp+tm) if tp+tm > 0 else 0
    dp_frac = dp/(dp+dm) if dp+dm > 0 else 0
    bal = tp*dp / (tm*dm) if tm*dm > 0 else 0
    print(f"{s:>3} {tp:>6.3f} {tm:>6.3f} {dp:>6.3f} {dm:>6.3f} {tp_frac:>7.3f} {dp_frac:>7.3f} {bal:>5.2f}")
