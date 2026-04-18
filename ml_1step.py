"""ML one-step Δmid prediction via xgboost, LOSO cross-validation.

Features per IDLE row (book state at time t):
  sp, mid-level encoded prices, queue counts at each level, sizes at L0,
  imbalances at L0/L1, deep-level summaries, session id (categorical).

Target: mid[t+1] - mid[t] (1 real step forward, one IDLE row ahead).

Training: leave-one-session-out. Report per-session test R² and compare to
Hawkes-sim baseline (+6.3% at K=200).
"""
import numpy as np, os, time

RECSZ = 54 * 4  # bytes per event record
STRIDE = 100    # subsample interval (match sim eval)

def load_session(s, offs, ev_mm):
    """Return (features_X, target_y) for session s sampled at stride=100.

    Feature row = book at IDLE[k*stride] for k=0..; target = mid change to IDLE[k*stride+1]."""
    lo_b, hi_b = int(offs[s]), int(offs[s+1])
    lo_r = lo_b // RECSZ; hi_r = hi_b // RECSZ
    block = ev_mm[lo_r:hi_r]
    idles_mask = (block[:, 0] == 8)
    idles = block[idles_mask]
    books = idles[:, 5:54].astype(np.float64)  # 49 cols per IDLE: aR, bR, aS, bS, aN, bN, y
    n = len(books)
    # seed indices: 0, stride, 2*stride, ...
    seeds = np.arange(0, n - 1, STRIDE)
    # Drop cases where seed+1 is out of range
    seeds = seeds[seeds + 1 < n]
    # Features at seed
    Xbooks = books[seeds]           # (N, 49)
    aR = Xbooks[:, 0:8];  bR = Xbooks[:, 8:16]
    aS = Xbooks[:, 16:24]; bS = Xbooks[:, 24:32]
    aN = Xbooks[:, 32:40]; bN = Xbooks[:, 40:48]
    mid = (aR[:, 0] + bR[:, 0]) / 2.0
    # Derived features
    sp = aR[:, 0] - bR[:, 0]
    imb0 = (aN[:, 0] - bN[:, 0]) / np.maximum(aN[:, 0] + bN[:, 0], 1)
    imb1 = (aN[:, 1] - bN[:, 1]) / np.maximum(aN[:, 1] + bN[:, 1], 1)
    imb_deep_a = aN[:, 1:].sum(axis=1); imb_deep_b = bN[:, 1:].sum(axis=1)
    imb_deep = (imb_deep_a - imb_deep_b) / np.maximum(imb_deep_a + imb_deep_b, 1)
    # Level spacings
    aR_gap = aR[:, 1:] - aR[:, :-1]  # (N, 7)
    bR_gap = bR[:, :-1] - bR[:, 1:]  # (N, 7)
    # Build feature matrix
    feats = [
        sp[:, None],
        imb0[:, None], imb1[:, None], imb_deep[:, None],
        aN[:, :3], bN[:, :3],
        aS[:, :1], bS[:, :1],
        aR_gap[:, :3], bR_gap[:, :3],
        (aN[:, 0] + bN[:, 0])[:, None],  # total top queue
    ]
    X = np.hstack(feats).astype(np.float32)
    # Target: mid[t+1] - mid[t]
    next_mid = (books[seeds + 1, 0] + books[seeds + 1, 8]) / 2.0
    y = (next_mid - mid).astype(np.float32)
    return X, y

offs = np.fromfile('data/sessions.events.raw', dtype=np.int64)
total_bytes = os.path.getsize('data/train.events')
ev_mm = np.memmap('data/train.events', dtype=np.int32, mode='r', shape=(total_bytes // RECSZ, 54))

print("loading sessions...")
t0 = time.time()
per_sess = []
for s in range(62):
    X, y = load_session(s, offs, ev_mm)
    per_sess.append((X, y))
print(f"  loaded in {time.time()-t0:.1f}s; per-session N avg = {np.mean([len(y) for _, y in per_sess]):.0f}")

from sklearn.ensemble import HistGradientBoostingRegressor

def xtx_r2(y_true, y_pred):
    sst = (y_true * y_true).sum()
    return 100 * (1 - ((y_pred - y_true) ** 2).sum() / sst) if sst > 0 else 0

# Full LOSO across all 62 sessions
print("\n=== full LOSO (62 folds) ===")
t0 = time.time()
r2_per_session = np.zeros(62)
# Pre-build full dataset for slicing
all_X = np.vstack([per_sess[s][0] for s in range(62)])
all_y = np.concatenate([per_sess[s][1] for s in range(62)])
sess_boundaries = np.cumsum([0] + [len(per_sess[s][1]) for s in range(62)])
print(f"  total dataset: {len(all_X):,} rows × {all_X.shape[1]} features")

for test_s in range(62):
    lo, hi = sess_boundaries[test_s], sess_boundaries[test_s+1]
    mask_train = np.ones(len(all_y), dtype=bool)
    mask_train[lo:hi] = False
    model = HistGradientBoostingRegressor(
        max_iter=100, max_depth=5, learning_rate=0.08,
        l2_regularization=1.0, min_samples_leaf=50
    )
    model.fit(all_X[mask_train], all_y[mask_train])
    yhat = model.predict(all_X[lo:hi])
    r2_per_session[test_s] = xtx_r2(all_y[lo:hi], yhat)
    if (test_s + 1) % 10 == 0:
        print(f"  ..{test_s+1}/62  elapsed {time.time()-t0:.1f}s  running mean R²={r2_per_session[:test_s+1].mean():+.3f}%")

print(f"\n=== ML LOSO results ===")
print(f"  mean R²    = {r2_per_session.mean():+.3f}%")
print(f"  median R²  = {np.median(r2_per_session):+.3f}%")
print(f"  ≥0 sess    = {(r2_per_session > 0).sum()}/62")
print(f"  top 5: " + ' '.join(f'ses{np.argsort(-r2_per_session)[i]}={r2_per_session[np.argsort(-r2_per_session)[i]]:+.2f}%' for i in range(5)))
print(f"  bot 5: " + ' '.join(f'ses{np.argsort(r2_per_session)[i]}={r2_per_session[np.argsort(r2_per_session)[i]]:+.2f}%' for i in range(5)))
print(f"  train+eval time: {time.time()-t0:.1f}s")

# Save per-session results for comparison
np.savetxt('/tmp/ml_r2.txt', r2_per_session, fmt='%.4f')
