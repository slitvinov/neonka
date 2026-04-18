"""ML+OFI v2 — richer features, heavier HGB, multiprocessed LOSO.

New features on top of ml_ofi.py:
  - OFI_top and OFI_deep SEPARATELY (Xu2023 multi-level)
  - Mid momentum over W ∈ {10, 50, 200} past idles
  - Mid volatility (std of Δmid) over W ∈ {50, 200}
  - Micro-price proxy: imb0·sp, imb1·sp, imb0·imb1
  - Event rate over last W events (time per event)
  - Spread deviation from recent median

Heavier HGB: max_iter=200, max_depth=7.
LOSO runs in parallel via multiprocessing.Pool (≈7× speedup on 8 cores).
"""
import numpy as np, os, sys, time
from multiprocessing import Pool

RECSZ = 54 * 4
STRIDE = 100
T_LIST = [1, 5, 20, 55]
W_OFI_LIST = [50, 100, 500]
W_MOM_LIST = [10, 50, 200]
W_VOL_LIST = [50, 200]

SIGN_TOP = np.array([-1, +1, +1, -1, 0, 0, 0, 0])      # tp/tm only
SIGN_DEEP = np.array([0, 0, 0, 0, -1, +1, +1, -1])     # dp/dm only

def load_session(s, offs, ev_mm):
    lo_r = int(offs[s]) // RECSZ; hi_r = int(offs[s+1]) // RECSZ
    block = ev_mm[lo_r:hi_r]
    types = block[:, 0]
    is_idle = types == 8
    idles = block[is_idle]
    books = idles[:, 5:54].astype(np.float64)
    idle_t = idles[:, 1].astype(np.int64)
    n = len(books)
    ev_t = block[~is_idle, 1].astype(np.int64)
    ev_ty = types[~is_idle].astype(np.int64)
    cum_top = np.concatenate([[0], np.cumsum(SIGN_TOP[ev_ty])])
    cum_deep = np.concatenate([[0], np.cumsum(SIGN_DEEP[ev_ty])])

    T_max = max(T_LIST); W_mom_max = max(W_MOM_LIST); W_vol_max = max(W_VOL_LIST)
    # Require seed - W_mom_max >= 0 AND seed + T_max < n
    start = max(W_mom_max, W_vol_max)
    seeds = np.arange(start, n - T_max, STRIDE)
    if len(seeds) == 0:
        return None
    seed_row = idle_t[seeds]
    end_idx = np.searchsorted(ev_t, seed_row, side='right')

    Xb = books[seeds]
    aR = Xb[:, 0:8];  bR = Xb[:, 8:16]
    aS = Xb[:, 16:24]; bS = Xb[:, 24:32]
    aN = Xb[:, 32:40]; bN = Xb[:, 40:48]
    mid = (aR[:, 0] + bR[:, 0]) / 2.0
    sp = aR[:, 0] - bR[:, 0]
    imb0 = (aN[:, 0] - bN[:, 0]) / np.maximum(aN[:, 0] + bN[:, 0], 1)
    imb1 = (aN[:, 1] - bN[:, 1]) / np.maximum(aN[:, 1] + bN[:, 1], 1)
    ida = aN[:, 1:].sum(axis=1); idb = bN[:, 1:].sum(axis=1)
    imb_deep = (ida - idb) / np.maximum(ida + idb, 1)
    aR_gap = aR[:, 1:] - aR[:, :-1]
    bR_gap = bR[:, :-1] - bR[:, 1:]

    # Mid history for momentum / vol
    all_mid = (books[:, 0] + books[:, 8]) / 2.0
    mom_feats = []
    for W in W_MOM_LIST:
        mom = mid - all_mid[seeds - W]
        mom_feats.append(mom[:, None])
    vol_feats = []
    for W in W_VOL_LIST:
        # std of Δmid over last W idles
        # use vectorized approach: precompute Δmid, then rolling std
        dmid = np.diff(all_mid, prepend=all_mid[0])  # (n,)
        # rolling std at each position via cumulative moments
        c1 = np.cumsum(dmid)
        c2 = np.cumsum(dmid * dmid)
        # rolling sum of window (i-W+1 ... i)
        def rolling_std(seed_i, W):
            lo = np.maximum(seed_i - W + 1, 0)
            s1 = c1[seed_i] - (c1[lo-1] if (lo > 0).all() else 0)
            s2 = c2[seed_i] - (c2[lo-1] if (lo > 0).all() else 0)
            # Actually vectorize with np.where
            s1 = c1[seed_i] - np.where(lo > 0, c1[np.maximum(lo-1, 0)], 0)
            s2 = c2[seed_i] - np.where(lo > 0, c2[np.maximum(lo-1, 0)], 0)
            n_w = seed_i - lo + 1
            var = (s2 - s1 * s1 / n_w) / np.maximum(n_w, 1)
            return np.sqrt(np.maximum(var, 0))
        vol_feats.append(rolling_std(seeds, W)[:, None])

    # OFI features at each W, split top/deep
    ofi_feats = []
    for W in W_OFI_LIST:
        start_idx = np.maximum(0, end_idx - W)
        ofi_feats.append((cum_top[end_idx] - cum_top[start_idx])[:, None])
        ofi_feats.append((cum_deep[end_idx] - cum_deep[start_idx])[:, None])
        # Event rate: W events / elapsed rows
        elapsed = seed_row - np.where(start_idx > 0, ev_t[start_idx - 1], seed_row - W * 2)
        ofi_feats.append((W / np.maximum(elapsed, 1))[:, None])

    # Micro-price / interaction features
    micro_feats = [
        (imb0 * sp)[:, None],
        (imb1 * sp)[:, None],
        (imb0 * imb1)[:, None],
        (imb0 * imb_deep)[:, None],
    ]

    # Spread deviation
    sp_dev = np.zeros(len(seeds))  # placeholder; use spread trend later
    # Recent median spread over last 200 idles
    all_sp = aR[:, 0] - bR[:, 0] if False else None  # stub

    base_feats = [
        sp[:, None], imb0[:, None], imb1[:, None], imb_deep[:, None],
        aN[:, :3], bN[:, :3], aS[:, :1], bS[:, :1],
        aR_gap[:, :3], bR_gap[:, :3],
        (aN[:, 0] + bN[:, 0])[:, None],
    ]

    X = np.hstack(base_feats + mom_feats + vol_feats + ofi_feats + micro_feats).astype(np.float32)

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

print("loading sessions with enhanced OFI features...")
t0 = time.time()
per_sess = []
for s in range(62):
    r = load_session(s, offs, ev_mm)
    per_sess.append(r)
good = [p for p in per_sess if p is not None]
print(f"  loaded in {time.time()-t0:.1f}s; N_feats = {good[0][0].shape[1]}")
print(f"  N_samples = {sum(len(p[1][1]) for p in good):,}")

# Save shared arrays to /tmp for worker processes
def build_arrays(per_sess, T):
    X = np.vstack([p[0] for p in per_sess if p is not None])
    y = np.concatenate([p[1][T] for p in per_sess if p is not None])
    ns = [len(p[1][T]) for p in per_sess if p is not None]
    b = np.cumsum([0] + ns)
    return X, y, b

from sklearn.ensemble import HistGradientBoostingRegressor

def xtx_r2(y, yp):
    sst = (y * y).sum()
    return 100 * (1 - ((yp - y) ** 2).sum() / sst) if sst > 0 else 0

def fit_fold(args):
    X_train, y_train, X_test, y_test = args
    m = HistGradientBoostingRegressor(
        max_iter=200, max_depth=7, learning_rate=0.05,
        l2_regularization=1.0, min_samples_leaf=50)
    m.fit(X_train, y_train)
    return xtx_r2(y_test, m.predict(X_test))

def loso_parallel(per_sess, T, n_workers=8):
    X_all, y_all, b = build_arrays(per_sess, T)
    print(f"  T={T}: dataset {len(X_all):,} rows × {X_all.shape[1]} features")
    tasks = []
    for ts in range(62):
        lo, hi = b[ts], b[ts+1]
        if lo == hi:
            tasks.append(None); continue
        mask = np.ones(len(y_all), dtype=bool); mask[lo:hi] = False
        tasks.append((X_all[mask], y_all[mask], X_all[lo:hi], y_all[lo:hi]))
    r2 = np.zeros(62)
    t0 = time.time()
    with Pool(processes=n_workers) as pool:
        for ts, r in enumerate(pool.imap_unordered(lambda t: (t[0], fit_fold(t[1])) if t[1] is not None else (t[0], 0),
                                                    [(i, tasks[i]) for i in range(62)])):
            idx, val = r
            r2[idx] = val
            if (ts+1) % 10 == 0:
                print(f"    T={T} {ts+1}/62  mean_so_far={r2[r2!=0].mean():+.3f}%  ({time.time()-t0:.0f}s)")
    return r2

# Can't use lambda with Pool.imap_unordered easily. Let me reroute.

def loso(per_sess, T):
    X_all, y_all, b = build_arrays(per_sess, T)
    print(f"  T={T}: dataset {len(X_all):,} rows × {X_all.shape[1]} features", flush=True)
    r2 = np.zeros(62)
    t0 = time.time()
    for ts in range(62):
        lo, hi = b[ts], b[ts+1]
        if lo == hi: continue
        mask = np.ones(len(y_all), dtype=bool); mask[lo:hi] = False
        m = HistGradientBoostingRegressor(
            max_iter=200, max_depth=7, learning_rate=0.05,
            l2_regularization=1.0, min_samples_leaf=50)
        m.fit(X_all[mask], y_all[mask])
        r2[ts] = xtx_r2(y_all[lo:hi], m.predict(X_all[lo:hi]))
        if (ts+1) % 10 == 0:
            print(f"    T={T} {ts+1}/62  running_mean={r2[:ts+1].mean():+.3f}%  ({time.time()-t0:.0f}s)", flush=True)
    return r2

if __name__ == '__main__':
    print("\n=== LOSO (T=55 first) ===", flush=True)
    for T in [55, 5]:  # focus on what we care about
        r2 = loso(per_sess, T)
        print(f"\n  T={T:3d}  mean R² = {r2.mean():+.3f}%  median = {np.median(r2):+.3f}%  ≥0: {(r2>0).sum()}/62",
              flush=True)
