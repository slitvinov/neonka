"""ML one-elementary-event Δmid prediction using train.events directly.

Each non-IDLE record = one training sample:
  features = book state at event (pre-event) + event type + level + distance
  target = mid change to the next IDLE record (end of current row transition)

For single-event rows: target is the mid change caused by this elementary event.
For multi-event rows: all events in a row share the same target (the row total),
but each event's features (type/level/distance) differ — the model learns which
event types move mid and by how much.

Cross-session LOSO with HistGradientBoostingRegressor.
"""
import numpy as np, os, time

RECSZ = 54 * 4
STRIDE = 200        # subsample events (many more than IDLEs)

def load_session_events(s, offs, ev_mm):
    lo_r = int(offs[s]) // RECSZ
    hi_r = int(offs[s+1]) // RECSZ
    block = ev_mm[lo_r:hi_r]
    # Iterate: for each non-IDLE record, find the NEXT IDLE record's mid
    types = block[:, 0]
    n = len(block)

    # Precompute mid per IDLE index (sparse array indexed by original position)
    is_idle = (types == 8)
    # For each record index i, find the next index j where is_idle[j] is True
    # (inclusive scan from right)
    next_idle_idx = np.full(n, -1, dtype=np.int64)
    nxt = -1
    for k in range(n - 1, -1, -1):
        if is_idle[k]: nxt = k
        next_idle_idx[k] = nxt

    # Gather samples from non-IDLE records
    event_positions = np.where(~is_idle)[0]
    # Skip events without a following IDLE
    event_positions = event_positions[next_idle_idx[event_positions] >= 0]
    # Subsample
    event_positions = event_positions[::STRIDE]
    if len(event_positions) == 0:
        return np.zeros((0, 22), dtype=np.float32), np.zeros(0, dtype=np.float32)

    # Feature extraction
    recs = block[event_positions]
    evt_type = recs[:, 0].astype(np.float64)
    evt_lvl  = recs[:, 2].astype(np.float64)
    evt_dist = recs[:, 3].astype(np.float64)
    books = recs[:, 5:54].astype(np.float64)
    aR, bR = books[:, 0:8], books[:, 8:16]
    aS, bS = books[:, 16:24], books[:, 24:32]
    aN, bN = books[:, 32:40], books[:, 40:48]
    mid_before = (aR[:, 0] + bR[:, 0]) / 2.0
    sp = aR[:, 0] - bR[:, 0]
    imb0 = (aN[:, 0] - bN[:, 0]) / np.maximum(aN[:, 0] + bN[:, 0], 1)
    imb1 = (aN[:, 1] - bN[:, 1]) / np.maximum(aN[:, 1] + bN[:, 1], 1)
    imb_d_a = aN[:, 1:].sum(axis=1); imb_d_b = bN[:, 1:].sum(axis=1)
    imb_deep = (imb_d_a - imb_d_b) / np.maximum(imb_d_a + imb_d_b, 1)
    aR_gap = aR[:, 1:] - aR[:, :-1]
    bR_gap = bR[:, :-1] - bR[:, 1:]

    feats = [
        evt_type[:, None], evt_lvl[:, None], evt_dist[:, None],
        sp[:, None], imb0[:, None], imb1[:, None], imb_deep[:, None],
        aN[:, :3], bN[:, :3],
        aS[:, :1], bS[:, :1],
        aR_gap[:, :3], bR_gap[:, :3],
        (aN[:, 0] + bN[:, 0])[:, None],
    ]
    X = np.hstack(feats).astype(np.float32)

    # Target: mid at next-IDLE row - mid at current event
    next_idles = block[next_idle_idx[event_positions]]
    mid_after = (next_idles[:, 5].astype(np.float64) + next_idles[:, 5 + 8]) / 2.0
    y = (mid_after - mid_before).astype(np.float32)
    return X, y

offs = np.fromfile('data/sessions.events.raw', dtype=np.int64)
total_bytes = os.path.getsize('data/train.events')
ev_mm = np.memmap('data/train.events', dtype=np.int32, mode='r', shape=(total_bytes // RECSZ, 54))

print("loading sessions (per elementary event)...")
t0 = time.time()
per_sess = []
for s in range(62):
    X, y = load_session_events(s, offs, ev_mm)
    per_sess.append((X, y))
n_tot = sum(len(y) for _, y in per_sess)
print(f"  loaded: {n_tot:,} elementary-event samples in {time.time()-t0:.1f}s")

from sklearn.ensemble import HistGradientBoostingRegressor

def xtx_r2(y_true, y_pred):
    sst = (y_true * y_true).sum()
    return 100 * (1 - ((y_pred - y_true) ** 2).sum() / sst) if sst > 0 else 0

print("\n=== LOSO across 62 sessions ===")
t0 = time.time()
all_X = np.vstack([per_sess[s][0] for s in range(62)])
all_y = np.concatenate([per_sess[s][1] for s in range(62)])
bounds = np.cumsum([0] + [len(per_sess[s][1]) for s in range(62)])
print(f"  dataset: {len(all_X):,} × {all_X.shape[1]} features")

r2 = np.zeros(62)
for ts in range(62):
    lo, hi = bounds[ts], bounds[ts+1]
    mask = np.ones(len(all_y), dtype=bool); mask[lo:hi] = False
    model = HistGradientBoostingRegressor(
        max_iter=100, max_depth=5, learning_rate=0.08,
        l2_regularization=1.0, min_samples_leaf=50)
    model.fit(all_X[mask], all_y[mask])
    r2[ts] = xtx_r2(all_y[lo:hi], model.predict(all_X[lo:hi]))
    if (ts+1) % 10 == 0:
        print(f"  ..{ts+1}/62  mean so far {r2[:ts+1].mean():+.3f}%  ({time.time()-t0:.1f}s)")

print(f"\n=== results ===")
print(f"  mean R²    = {r2.mean():+.3f}%")
print(f"  median R²  = {np.median(r2):+.3f}%")
print(f"  ≥0 sess    = {(r2 > 0).sum()}/62")
print(f"  top 5: " + ' '.join(f'ses{np.argsort(-r2)[i]}={r2[np.argsort(-r2)[i]]:+.2f}%' for i in range(5)))
print(f"  bot 5: " + ' '.join(f'ses{np.argsort(r2)[i]}={r2[np.argsort(r2)[i]]:+.2f}%' for i in range(5)))
print(f"  ses45={r2[45]:+.3f}%  ses55={r2[55]:+.3f}%  ses53={r2[53]:+.3f}%")
print(f"\n  total time: {time.time()-t0:.1f}s")
