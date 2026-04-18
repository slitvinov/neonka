import sys, numpy as np, os

r = np.fromfile('data/train.raw', dtype=np.int32).reshape(-1, 49)
b = np.fromfile('data/sessions.raw', dtype=np.int64)
W_WIN     = 50
LAG       = 50
ALPHA_DEF = 0.25
OUT_PATH  = 'data/tables/hawkes_params.txt'

lines = []
for S in range(len(b) - 1):
    lo, hi = int(b[S]), int(b[S+1])
    x = r[lo:hi]
    if len(x) < 1000:
        lines.append((S, ALPHA_DEF, 0.02))
        continue
    chg = np.any(x[1:, :48] != x[:-1, :48], axis=1).astype(float)
    if len(chg) < W_WIN + LAG + 100:
        lines.append((S, ALPHA_DEF, 0.02))
        continue
    rate = np.convolve(chg, np.ones(W_WIN)/W_WIN, mode='valid')
    r0 = rate - rate.mean(); v = (r0*r0).mean()
    if v <= 0:
        lines.append((S, ALPHA_DEF, 0.02))
        continue
    acf_vals = []
    for lag in (LAG, LAG * 2, LAG * 4):
        if len(r0) > lag + 10:
            acf_vals.append(max(0.01, min(0.95, float((r0[:-lag]*r0[lag:]).mean()/v))))
    if not acf_vals:
        lines.append((S, ALPHA_DEF, 0.02))
        continue
    betas_eff = [-np.log(a) / lag for a, lag in zip(acf_vals, (LAG, LAG*2, LAG*4))]
    beta_eff = np.median(betas_eff)
    alpha = ALPHA_DEF
    beta = beta_eff / (1 - alpha)
    beta = max(0.001, min(0.1, beta))
    lines.append((S, alpha, beta))

with open(OUT_PATH, 'w') as f:
    for S, a, bta in lines:
        f.write(f"{S:02d} {a:.4f} {bta:.5f}\n")

print(f"wrote {len(lines)} sessions to {OUT_PATH}")
print(f"{'ses':>4}  {'α':>6}  {'β':>7}")
for S, a, bta in lines[:8]:
    print(f"{S:>4}  {a:>6.4f}  {bta:>7.5f}")
print("...")
alphas = np.array([l[1] for l in lines]); betas = np.array([l[2] for l in lines])
print(f"β: mean={betas.mean():.5f}  median={np.median(betas):.5f}  min={betas.min():.5f}  max={betas.max():.5f}")
