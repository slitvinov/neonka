"""ML Δmid prediction with OFI features, LOSO across 62 sessions.

Features per IDLE seed (book at time t):
  sp, imb0, imb1, imb_deep, aN[0..2], bN[0..2], aS[0], bS[0], aR_gap[0..2], bR_gap[0..2],
  top_queue_sum,
  + OFI_top(W=50/100/500) — signed event-count windows (CKS 2014)
  + OFI_multi(W=50/100/500) — OFI extended to deep levels

Target: mid[t + T] - mid[t].  Sweeps T ∈ {1, 5, 20, 55}.
LOSO with HistGradientBoostingRegressor.
"""
import numpy as np, os, sys, time

RECSZ = 54 * 4
STRIDE = 100
T_LIST = [1, 5, 20, 55]
W_LIST = [50, 100, 500]

# Top-level OFI: tp_a=-1 tp_b=+1 tm_a=+1 tm_b=-1 (dp/dm=0)
SIGN_TOP = np.array([-1, +1, +1, -1, 0, 0, 0, 0])
# Multi-level OFI (includes deep): dp_a=-1 dp_b=+1 dm_a=+1 dm_b=-1
SIGN_MULTI = np.array([-1, +1, +1, -1, -1, +1, +1, -1])

def load_session(s, offs, ev_mm):
    lo_r = int(offs[s]) // RECSZ; hi_r = int(offs[s+1]) // RECSZ
    block = ev_mm[lo_r:hi_r]
    types = block[:, 0]
    is_idle = types == 8
    idles = block[is_idle]
    books = idles[:, 5:54].astype(np.float64)
    idle_t = idles[:, 1].astype(np.int64)
    n = len(books)
    # Non-IDLE events with times
    ev_t = block[~is_idle, 1].astype(np.int64)
    ev_ty = types[~is_idle].astype(np.int64)
    cum_top = np.concatenate([[0], np.cumsum(SIGN_TOP[ev_ty])])
    cum_mul = np.concatenate([[0], np.cumsum(SIGN_MULTI[ev_ty])])

    # Seeds: every STRIDE idles that can look ahead at least T=max
    T_max = max(T_LIST)
    seeds = np.arange(0, n - T_max, STRIDE)
    if len(seeds) == 0:
        return None
    seed_row = idle_t[seeds]
    end_idx = np.searchsorted(ev_t, seed_row, side='right')

    Xb = books[seeds]
    aR = Xb[:, 0:8];  bR = Xb[:, 8:16]
    aS = Xb[:, 16:24]; bS = Xb[:, 24:32]
    aN = Xb[:, 32:40]; bN = Xb[:, 40:48]
    sp = aR[:, 0] - bR[:, 0]
    imb0 = (aN[:, 0] - bN[:, 0]) / np.maximum(aN[:, 0] + bN[:, 0], 1)
    imb1 = (aN[:, 1] - bN[:, 1]) / np.maximum(aN[:, 1] + bN[:, 1], 1)
    ida = aN[:, 1:].sum(axis=1); idb = bN[:, 1:].sum(axis=1)
    imb_deep = (ida - idb) / np.maximum(ida + idb, 1)
    aR_gap = aR[:, 1:] - aR[:, :-1]
    bR_gap = bR[:, :-1] - bR[:, 1:]

    base_feats = [
        sp[:, None], imb0[:, None], imb1[:, None], imb_deep[:, None],
        aN[:, :3], bN[:, :3], aS[:, :1], bS[:, :1],
        aR_gap[:, :3], bR_gap[:, :3],
        (aN[:, 0] + bN[:, 0])[:, None],
    ]

    # OFI features at each W
    ofi_feats = []
    for W in W_LIST:
        start_idx = np.maximum(0, end_idx - W)
        ofi_feats.append((cum_top[end_idx] - cum_top[start_idx])[:, None])
        ofi_feats.append((cum_mul[end_idx] - cum_mul[start_idx])[:, None])

    X = np.hstack(base_feats + ofi_feats).astype(np.float32)

    # Targets for each T
    mid = (aR[:, 0] + bR[:, 0]) / 2.0
    ys = {}
    for T in T_LIST:
        fut = seeds + T
        fut_mid = (books[fut, 0] + books[fut, 8]) / 2.0
        ys[T] = (fut_mid - mid).astype(np.float32)
    return X, ys

offs = np.fromfile('data/sessions.events.raw', dtype=np.int64)
total_bytes = os.path.getsize('data/train.events')
ev_mm = np.memmap('data/train.events', dtype=np.int32, mode='r',
                  shape=(total_bytes // RECSZ, 54))

print("loading sessions with OFI features...")
t0 = time.time()
per_sess = []
for s in range(62):
    r = load_session(s, offs, ev_mm)
    if r is None:
        per_sess.append(None); continue
    per_sess.append(r)
print(f"  loaded in {time.time()-t0:.1f}s; N_feats = {per_sess[0][0].shape[1]}")

from sklearn.ensemble import HistGradientBoostingRegressor

def xtx_r2(y, yp):
    sst = (y * y).sum()
    return 100 * (1 - ((yp - y) ** 2).sum() / sst) if sst > 0 else 0

def loso(per_sess, T):
    all_X = np.vstack([p[0] for p in per_sess if p is not None])
    all_y = np.concatenate([p[1][T] for p in per_sess if p is not None])
    ns = [len(p[1][T]) for p in per_sess if p is not None]
    b = np.cumsum([0] + ns)
    r2 = np.zeros(62)
    t0 = time.time()
    for ts in range(62):
        lo, hi = b[ts], b[ts+1]
        mask = np.ones(len(all_y), dtype=bool); mask[lo:hi] = False
        m = HistGradientBoostingRegressor(max_iter=100, max_depth=5,
                learning_rate=0.08, l2_regularization=1.0, min_samples_leaf=50)
        m.fit(all_X[mask], all_y[mask])
        r2[ts] = xtx_r2(all_y[lo:hi], m.predict(all_X[lo:hi]))
        if (ts+1) % 20 == 0:
            print(f"    T={T} ..{ts+1}/62  mean={r2[:ts+1].mean():+.3f}%  ({time.time()-t0:.0f}s)")
    return r2

print("\n=== LOSO (62 folds) with OFI features ===")
print(f"{'T':>3} {'mean R²':>10} {'median':>10} {'>=0 count':>10}")
for T in T_LIST:
    r2 = loso(per_sess, T)
    print(f"{T:>3} {r2.mean():>+9.3f}% {np.median(r2):>+9.3f}% {(r2>0).sum():>6d}/62")
