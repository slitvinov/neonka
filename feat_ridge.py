import sys, time
import numpy as np

t0 = time.time()
r = np.fromfile('data/train.raw', dtype=np.int32).reshape(-1, 49)
b = np.fromfile('data/sessions.raw', dtype=np.int64)
print(f"loaded {len(r):,} rows, {len(b)-1} sessions in {time.time()-t0:.1f}s", flush=True)

ALPHA = 0.02      # EMA decay (half-life ~35 ticks)
LAM   = 1.0       # Ridge regularization

def ema_np(x, alpha):
    out = np.empty_like(x, dtype=np.float32)
    m = 0.0
    for i in range(len(x)):
        m = alpha * x[i] + (1 - alpha) * m
        out[i] = m
    return out

def compute_features(x):
    aR0 = x[:, 0].astype(np.float32);  bR0 = x[:, 8].astype(np.float32)
    aN0 = x[:, 32].astype(np.float32); bN0 = x[:, 40].astype(np.float32)
    aN1 = x[:, 33].astype(np.float32); bN1 = x[:, 41].astype(np.float32)
    aN2 = x[:, 34].astype(np.float32); bN2 = x[:, 42].astype(np.float32)
    aS0 = x[:, 16].astype(np.float32); bS0 = x[:, 24].astype(np.float32)
    sp   = aR0 - bR0
    mid  = (aR0 + bR0) / 2
    imb0 = (aN0 - bN0) / np.maximum(aN0 + bN0, 1)
    imb1 = (aN1 - bN1) / np.maximum(aN1 + bN1, 1)
    imb2 = (aN2 - bN2) / np.maximum(aN2 + bN2, 1)
    imb_sz = (aS0 - bS0) / np.maximum(aS0 + bS0, 1)
    micro  = -sp * imb0 / 2
    dm = np.diff(mid, prepend=mid[0])
    daN = np.diff(aN0, prepend=aN0[0])
    dbN = np.diff(bN0, prepend=bN0[0])
    ofi_raw = dbN - daN
    chg = (np.abs(np.diff(x[:, :48], axis=0, prepend=x[:1, :48])).sum(axis=1) > 0).astype(np.float32)
    act_ema = ema_np(chg, ALPHA)
    vol_ema = ema_np(np.abs(dm), ALPHA)
    imb_ema = ema_np(imb0, ALPHA)
    sp_ema  = ema_np(sp, ALPHA)
    ofi_ema = ema_np(ofi_raw, ALPHA)
    return np.stack([
        sp, imb0, imb1, imb2, imb_sz,
        np.log1p(aN0), np.log1p(bN0), np.log1p(aN1), np.log1p(bN1),
        micro, imb0*sp, imb0**2, np.abs(imb0),
        dm, act_ema, vol_ema, imb_ema, sp_ema, ofi_ema,
        imb0 - imb_ema, sp - sp_ema,
    ], axis=1).astype(np.float32)

# Build all features + target y
all_F = []; all_Y = []; all_S = []
for s in range(len(b) - 1):
    lo, hi = int(b[s]), int(b[s+1])
    x = r[lo:hi]
    if len(x) < 100: continue
    F = compute_features(x)
    y = x[:, 48].astype(np.float32) / 4.0
    all_F.append(F); all_Y.append(y); all_S.append(np.full(len(F), s, dtype=np.int32))

F_all = np.concatenate(all_F); Y_all = np.concatenate(all_Y); S_all = np.concatenate(all_S)
print(f"pooled {len(F_all):,} samples, {F_all.shape[1]} features in {time.time()-t0:.1f}s", flush=True)

def xtx_r2(y, yh): return 1 - ((yh - y)**2).sum() / (y**2).sum()
def std_r2(y, yh): return 1 - ((yh - y)**2).sum() / ((y - y.mean())**2).sum()

def ridge_fit_eval(F_tr, y_tr, F_te, y_te, lam=LAM):
    mu = F_tr.mean(0); sd = F_tr.std(0) + 1e-8
    Ft = (F_tr - mu) / sd
    Fe = (F_te - mu) / sd
    A = np.hstack([Ft, np.ones((len(Ft), 1), dtype=np.float32)])
    XtX = A.T.astype(np.float64) @ A.astype(np.float64)
    w = np.linalg.solve(XtX + lam * np.eye(A.shape[1]), A.T.astype(np.float64) @ y_tr.astype(np.float64))
    yh = np.hstack([Fe, np.ones((len(Fe), 1), dtype=np.float32)]).astype(np.float64) @ w
    return xtx_r2(y_te, yh), std_r2(y_te, yh)

# 1) Last-N-sessions OOS split (train 0..51, test 52..62)
print("\n--- 1) Last-11-sessions time-split test ---")
tr_mask = S_all < 52
te_mask = S_all >= 52
x_r2, s_r2 = ridge_fit_eval(F_all[tr_mask], Y_all[tr_mask], F_all[te_mask], Y_all[te_mask])
print(f"  train={tr_mask.sum():,}  test={te_mask.sum():,}  XTX R²={100*x_r2:+.3f}%  std R²={100*s_r2:+.3f}%")

# 2) LOSO CV
print("\n--- 2) Leave-one-session-out CV ---")
r2s_x = []; r2s_s = []; ns = []
for s in range(len(b) - 1):
    mask = S_all != s
    test_mask = ~mask
    if test_mask.sum() < 100: continue
    x_r2, s_r2 = ridge_fit_eval(F_all[mask], Y_all[mask], F_all[test_mask], Y_all[test_mask])
    r2s_x.append(x_r2); r2s_s.append(s_r2); ns.append(test_mask.sum())
print(f"  mean XTX R²={100*np.mean(r2s_x):+.3f}%  weighted={100*np.average(r2s_x, weights=ns):+.3f}%  "
      f"median={100*np.median(r2s_x):+.3f}%  neg={sum(x<0 for x in r2s_x)}/63")
print(f"  mean std R²={100*np.mean(r2s_s):+.3f}%")

# 3) Session 50/50 time-split
print("\n--- 3) Per-session 50/50 time-split mean ---")
r2s_x = []; ns = []
for s in range(len(b) - 1):
    mask = S_all == s
    idx = np.where(mask)[0]
    if len(idx) < 100: continue
    half = len(idx) // 2
    tr, te = idx[:half], idx[half:]
    x_r2, _ = ridge_fit_eval(F_all[tr], Y_all[tr], F_all[te], Y_all[te])
    r2s_x.append(x_r2); ns.append(len(te))
print(f"  mean XTX R²={100*np.mean(r2s_x):+.3f}%  weighted={100*np.average(r2s_x, weights=ns):+.3f}%  "
      f"neg={sum(x<0 for x in r2s_x)}/63")

print(f"\ntotal time: {time.time()-t0:.1f}s")
