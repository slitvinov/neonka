"""Compute synthetic-data evidence statistics and write .dat files for gnuplot.

Writes to sim/data/:
  dmid_pooled.dat     — pooled P(Δmid=k) for k in [-10, 10], sym candidate
  dmid_per_sess.dat   — per-session log2(P(+1)/P(-1)) and log2(P(+2)/P(-2))
  event_balance.dat   — per-session bid/ask event-count ratios
  fingerprints.dat    — per-session vol, spread, ev_rate, mid_return, drift
  sym_pooled_ratio.dat— pooled log2(P(+k)/P(-k)) for k in 1..10 with z-scores
"""
import numpy as np, os, sys

RECSZ = 216
OUT = os.path.dirname(os.path.abspath(__file__)) + '/data'
os.makedirs(OUT, exist_ok=True)

ev = np.memmap('data/train.events', dtype=np.int32, mode='r').reshape(-1, 54)
offs = np.fromfile('data/sessions.events.raw', dtype=np.int64)

# 1) Pooled Δmid distribution & per-session log2 ratios
P = np.zeros((62, 21))    # P(Δ=-10)..P(Δ=+10)
N_per_sess = np.zeros(62, dtype=np.int64)
type_counts = np.zeros((62, 8), dtype=np.int64)
mid_start = np.zeros(62); mid_end = np.zeros(62)
vol = np.zeros(62); sp_med = np.zeros(62); imb_mean = np.zeros(62); ev_rate = np.zeros(62)
n_idle = np.zeros(62, dtype=np.int64)
abs_counts = np.zeros((62, 21), dtype=np.int64)

for s in range(62):
    lo = int(offs[s])//RECSZ; hi = int(offs[s+1])//RECSZ
    block = ev[lo:hi]
    types = block[:, 0]
    is_idle = types == 8
    idles = block[is_idle]
    for t in range(8):
        type_counts[s, t] = (types == t).sum()
    mid = (idles[:, 5].astype(np.float64) + idles[:, 5+8]) / 2.0
    dmid = np.diff(mid)
    N_per_sess[s] = len(dmid)
    n_idle[s] = len(idles)
    for i, k in enumerate(range(-10, 11)):
        c = (dmid == k).sum()
        abs_counts[s, i] = c
        P[s, i] = c / max(len(dmid), 1)
    mid_start[s] = mid[0]; mid_end[s] = mid[-1]
    vol[s] = dmid.std()
    sp = (idles[:, 5] - idles[:, 5+8]).astype(np.int64)
    sp_med[s] = float(np.median(sp))
    aN0 = idles[:, 37].astype(np.float64); bN0 = idles[:, 45].astype(np.float64)
    imb_mean[s] = ((aN0 - bN0) / np.maximum(aN0 + bN0, 1)).mean()
    ev_rate[s] = (~is_idle).sum() / max(len(idles), 1)

# --- Pooled Δmid histogram (log-y) ---
pooled = abs_counts.sum(axis=0)
total = pooled.sum()
with open(f'{OUT}/dmid_pooled.dat', 'w') as f:
    f.write("# k  count  P(Δ=k)  log10_P\n")
    for i, k in enumerate(range(-10, 11)):
        p = pooled[i] / max(total, 1)
        f.write(f"{k}\t{pooled[i]}\t{p:.8e}\t{np.log10(max(p, 1e-12)):+.6f}\n")
print(f"wrote dmid_pooled.dat: pooled N={total}")

# --- Per-session log2(P+/P-) ---
with open(f'{OUT}/dmid_per_sess.dat', 'w') as f:
    f.write("# s  log2(P+1/P-1)  log2(P+2/P-2)  log2(P+3/P-3)  N_dmid  P+1_N  P-1_N\n")
    for s in range(62):
        r1 = np.log2(P[s, 11] / P[s, 9]) if P[s, 9] > 0 else 0
        r2 = np.log2(P[s, 12] / P[s, 8]) if P[s, 8] > 0 else 0
        r3 = np.log2(P[s, 13] / P[s, 7]) if P[s, 7] > 0 else 0
        f.write(f"{s}\t{r1:+.6f}\t{r2:+.6f}\t{r3:+.6f}\t{N_per_sess[s]}\t{abs_counts[s, 11]}\t{abs_counts[s, 9]}\n")
print("wrote dmid_per_sess.dat")

# --- Pooled log2(P+k/P-k) with z-scores vs 50/50 null ---
with open(f'{OUT}/sym_pooled_ratio.dat', 'w') as f:
    f.write("# k  N+k  N-k  log2_ratio  z_vs_5050\n")
    for k in range(1, 11):
        i_plus = 10 + k; i_minus = 10 - k
        np_cnt = pooled[i_plus]; nm_cnt = pooled[i_minus]
        tot = np_cnt + nm_cnt
        if tot == 0 or nm_cnt == 0:
            continue
        log2r = np.log2(np_cnt / nm_cnt)
        z = (np_cnt - tot/2) / np.sqrt(tot/4)
        f.write(f"{k}\t{np_cnt}\t{nm_cnt}\t{log2r:+.8f}\t{z:+.4f}\n")
print("wrote sym_pooled_ratio.dat")

# --- Event balance (bid vs ask type counts per session) ---
with open(f'{OUT}/event_balance.dat', 'w') as f:
    f.write("# s  log2(tp_a/tp_b)  log2(tm_a/tm_b)  log2(dp_a/dp_b)  log2(dm_a/dm_b)  "
            "log2((tp_a+tm_a)/(tp_b+tm_b)) log2((dp_a+dm_a)/(dp_b+dm_b))\n")
    for s in range(62):
        tc = type_counts[s]
        def lr(a, b): return np.log2(a / b) if a > 0 and b > 0 else 0
        f.write(f"{s}\t"
                f"{lr(tc[0], tc[1]):+.6f}\t{lr(tc[2], tc[3]):+.6f}\t"
                f"{lr(tc[4], tc[5]):+.6f}\t{lr(tc[6], tc[7]):+.6f}\t"
                f"{lr(tc[0]+tc[2], tc[1]+tc[3]):+.6f}\t{lr(tc[4]+tc[6], tc[5]+tc[7]):+.6f}\n")
print("wrote event_balance.dat")

# --- SMOKING GUN: cumulative bid/ask event counts over session time ---
# For each event type (tp, tm, dp, dm), plot N_cum_ask(t) and N_cum_bid(t) vs row-time.
# If synthetic-symmetric, the two curves overlap tick-by-tick. Real data never does.
# Dump ses45 (mid-typical session) at downsampled resolution.
os.makedirs(f'{OUT}', exist_ok=True)
SAMPLE_SES = [0, 30, 45, 56]       # a few representative sessions
DOWN = 500                          # row-downsampling for plotting
for s in SAMPLE_SES:
    lo = int(offs[s])//RECSZ; hi = int(offs[s+1])//RECSZ
    block = ev[lo:hi]
    types_ = block[:, 0].astype(np.int64)
    row = block[:, 1].astype(np.int64)
    # per-type indicator streams over event records (NOT IDLEs)
    # For each event index, use type as 0..7
    # Cumulative at row-time: bin events by row with cumsum
    n_rows = int(row.max()) + 1
    cum = np.zeros((8, n_rows + 1), dtype=np.int64)
    for t in range(8):
        mask = types_ == t
        ev_rows = row[mask]
        # Count events per row via bincount
        c = np.bincount(ev_rows, minlength=n_rows + 1)
        cum[t] = np.cumsum(c)
    with open(f'{OUT}/cum_ses{s}.dat', 'w') as f:
        f.write("# row\ttp_a\ttp_b\ttm_a\ttm_b\tdp_a\tdp_b\tdm_a\tdm_b\t"
                "diff_tp\tdiff_tm\tdiff_dp\tdiff_dm\n")
        for r in range(0, n_rows + 1, DOWN):
            d_tp = cum[0, r] - cum[1, r]
            d_tm = cum[2, r] - cum[3, r]
            d_dp = cum[4, r] - cum[5, r]
            d_dm = cum[6, r] - cum[7, r]
            f.write(f"{r}\t{cum[0,r]}\t{cum[1,r]}\t{cum[2,r]}\t{cum[3,r]}\t"
                    f"{cum[4,r]}\t{cum[5,r]}\t{cum[6,r]}\t{cum[7,r]}\t"
                    f"{d_tp}\t{d_tm}\t{d_dp}\t{d_dm}\n")
    print(f"wrote cum_ses{s}.dat  (rows: {n_rows})")

# Per-session: max absolute gap between cumulative a-side and b-side for each event type
# Summary: is the gap bounded or does it drift?
with open(f'{OUT}/cum_gap_summary.dat', 'w') as f:
    f.write("# s  max_tp_diff  max_tm_diff  max_dp_diff  max_dm_diff  "
            "end_tp_diff  end_tm_diff  end_dp_diff  end_dm_diff\n")
    for s in range(62):
        lo = int(offs[s])//RECSZ; hi = int(offs[s+1])//RECSZ
        block = ev[lo:hi]
        types_ = block[:, 0].astype(np.int64)
        # Event ORDER in the block (not row time) is enough: count cumulative signed balance
        # as events are encountered. max|balance| over the sequence, end balance.
        max_abs = np.zeros(4, dtype=np.int64)
        end_val = np.zeros(4, dtype=np.int64)
        for pair, (a, b) in enumerate([(0,1), (2,3), (4,5), (6,7)]):
            sign = np.zeros(len(types_), dtype=np.int64)
            sign[types_ == a] = 1
            sign[types_ == b] = -1
            bal = np.cumsum(sign)
            if len(bal) == 0:
                max_abs[pair] = 0; end_val[pair] = 0
            else:
                max_abs[pair] = int(np.abs(bal).max())
                end_val[pair] = int(bal[-1])
        f.write(f"{s}\t{max_abs[0]}\t{max_abs[1]}\t{max_abs[2]}\t{max_abs[3]}\t"
                f"{end_val[0]}\t{end_val[1]}\t{end_val[2]}\t{end_val[3]}\n")
        f.flush()
print("wrote cum_gap_summary.dat")

# --- Fingerprints per session ---
with open(f'{OUT}/fingerprints.dat', 'w') as f:
    f.write("# s  n_idle  ev_rate  mid_start  mid_end  mid_return  vol_dmid  sp_med  imb_mean\n")
    for s in range(62):
        f.write(f"{s}\t{n_idle[s]}\t{ev_rate[s]:.4f}\t"
                f"{mid_start[s]:.0f}\t{mid_end[s]:.0f}\t{mid_end[s]-mid_start[s]:+.0f}\t"
                f"{vol[s]:.4f}\t{sp_med[s]:.1f}\t{imb_mean[s]:+.4f}\n")
print("wrote fingerprints.dat")

# --- Summary to stdout ---
print("\n=== HEADLINE EVIDENCE ===")
print(f"Pooled P(Δ=+1) = {P[:, 11].mean():.6f}   P(Δ=-1) = {P[:, 9].mean():.6f}")
print(f"  ratio +1/-1 = {P[:, 11].mean()/P[:, 9].mean():.6f}")
p1p = pooled[11]; p1m = pooled[9]
tot1 = p1p + p1m
print(f"  pooled counts: +1: {p1p}  -1: {p1m}  z-score = {(p1p-tot1/2)/np.sqrt(tot1/4):+.3f}")
r1_arr = np.log2(P[:, 11] / np.maximum(P[:, 9], 1e-12))
print(f"  log2(P+1/P-1) across 62 sess: mean={r1_arr.mean():+.5f}  std={r1_arr.std():.5f}")
theo = 2 * np.sqrt(0.088*0.912/np.median(N_per_sess)) / (0.088 * np.log(2))
print(f"  expected std if 50/50: {theo:.5f}  — ratio obs/theo = {r1_arr.std()/theo:.2f}")
