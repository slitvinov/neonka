"""Structural anomaly scan on train.raw.
Probes: session boundaries, intra-session time structure, price clustering,
size quirks, rate bursts, spread regimes, and the big-dip region we saw.
"""
import numpy as np
import sys

raw = np.fromfile("data/train.raw", dtype=np.int32).reshape(-1, 49)
sess = np.fromfile("data/sessions.raw", dtype=np.int64)
n = len(raw)
print(f"rows {n:,}  cols 49  sessions {len(sess)-1}")

aR0 = raw[:,0]; aS0 = raw[:,16]; aN0 = raw[:,32]
bR0 = raw[:,8]; bS0 = raw[:,24]; bN0 = raw[:,40]
mid = (aR0 + bR0) / 2.0
sp = aR0 - bR0

print("\n=== 1. Session size anomalies ===")
lens = np.diff(sess)
print(f"  session len: min={lens.min():,} max={lens.max():,} median={int(np.median(lens)):,}")
print(f"  top 5 longest:  {sorted(range(63), key=lambda s:-lens[s])[:5]}")
print(f"  top 5 shortest: {sorted(range(63), key=lambda s: lens[s])[:5]}")

print("\n=== 2. Boundary price jumps (overnight gaps) ===")
gaps = []
for i in range(1, len(sess)-1):
    left = mid[sess[i]-1]
    right = mid[sess[i]]
    gaps.append(right - left)
gaps = np.array(gaps)
print(f"  n={len(gaps)} mean={gaps.mean():+.1f} std={gaps.std():.1f} |max|={np.abs(gaps).max():.1f}")
print(f"  top 5 largest |gap| sessions (the next sess idx):")
order = np.argsort(-np.abs(gaps))
for i in order[:5]:
    print(f"    ses {i+1:>3}  gap={gaps[i]:+7.1f}  ({mid[sess[i]-1]:.1f} -> {mid[sess[i]]:.1f})")

print("\n=== 3. Intra-session activity shape (open/close/lunch) ===")
buckets = 10
rate_by_bucket = np.zeros(buckets)
for s in range(63):
    a, b = sess[s], sess[s+1]
    L = b - a
    # proxy activity: how many ticks have non-zero event = (mid changes or sizes change)
    mid_s = mid[a:b]
    mv = (np.diff(mid_s) != 0).astype(float)
    idx = (np.arange(L-1) * buckets // (L-1))
    for k in range(buckets):
        rate_by_bucket[k] += mv[idx==k].mean() if (idx==k).sum() > 0 else 0
rate_by_bucket /= 63
print(f"  bucket:  {' '.join(f'{i:>5d}' for i in range(buckets))}")
print(f"  P(move): {' '.join(f'{v:>.3f}' for v in rate_by_bucket)}")
shape = rate_by_bucket / rate_by_bucket.mean()
print(f"  relative: {' '.join(f'{v:>.3f}' for v in shape)}")

print("\n=== 4. Price level clustering (round-number magnet?) ===")
ends = (aR0 % 40)
uq, cnt = np.unique(ends, return_counts=True)
exp = n / len(uq)
print(f"  ask-price mod-40 (= mod £1.0 since 1 real unit=4 raw):")
top = np.argsort(-cnt)[:8]
for i in top:
    print(f"    ends={uq[i]:>3}  count={cnt[i]:>11,}  ratio_to_uniform={cnt[i]/exp:.3f}")

print("\n  bid-price mod-4 (= tick boundary):")
ends2 = bR0 % 4
uq2, cnt2 = np.unique(ends2, return_counts=True)
for u, c in zip(uq2, cnt2): print(f"    {u} -> {c:>12,} ({c/n*100:.2f}%)")

print("\n=== 5. Size distribution quirks (round lots?) ===")
uq, cnt = np.unique(aS0, return_counts=True)
print(f"  ask_size[0]: distinct={len(uq)} max={aS0.max()}")
for v, c in sorted(zip(uq[:12], cnt[:12])):
    print(f"    size={v:>3}  count={c:>12,} ({c/n*100:.2f}%)")
print(f"  round-10 ratio: {((aS0 % 10)==0).mean()*100:.2f}%  vs uniform 10%")
print(f"  round-100 ratio: {((aS0 % 100)==0).mean()*100:.2f}%  vs uniform 1%")

print("\n=== 6. Spread regime shifts ===")
print(f"  sp dist: median={int(np.median(sp))} P95={int(np.percentile(sp,95))} max={sp.max()}")
for s in range(63):
    a, b = sess[s], sess[s+1]
    sm = int(np.median(sp[a:b]))
    if sm > 12 or sm < 4:
        print(f"    ses {s:>2}  median_sp={sm}")

print("\n=== 7. Big-drop region (sessions around ~52-56) ===")
for s in range(48, 60):
    a, b = sess[s], sess[s+1]
    mid_s = mid[a:b]
    drop = mid_s.max() - mid_s.min()
    print(f"  ses {s:>2}  len={b-a:>7,}  mid: start={mid_s[0]:.0f} min={mid_s.min():.0f} max={mid_s.max():.0f} end={mid_s[-1]:.0f}  swing={drop:.0f}")

print("\n=== 8. Zero-event runs (quiet periods) ===")
mv = np.diff(mid) != 0
# run length of false (no move)
changes = np.flatnonzero(mv)
gaps = np.diff(changes)
print(f"  n-gaps {len(gaps)}  mean-quiet={gaps.mean():.1f}  max-quiet={gaps.max()}")
print(f"  quiet >= 1000 ticks: {(gaps>=1000).sum()}")
print(f"  quiet >= 5000 ticks: {(gaps>=5000).sum()}")

print("\n=== 9. Events-per-session count check ===")
totevt = []
for s in range(63):
    a, b = sess[s], sess[s+1]
    mv_s = np.diff(mid[a:b]) != 0
    totevt.append(mv_s.sum())
totevt = np.array(totevt)
print(f"  mid-moves/session: mean={totevt.mean():.0f}  cv={totevt.std()/totevt.mean()*100:.1f}%")
print(f"  rate per 1000 ticks: {(totevt / lens * 1000).mean():.1f} ± {(totevt / lens * 1000).std():.1f}")

print("\n=== 10. Imbalance persistence at boundaries ===")
imb0 = (aN0.astype(np.int64) - bN0) / (aN0 + bN0).clip(1)
bnd_imb = [imb0[sess[s]] for s in range(63)]
print(f"  imb at session start: mean={np.mean(bnd_imb):+.3f} std={np.std(bnd_imb):.3f}")
print(f"  correlation(imb_start, gap_to_prev): {np.corrcoef(bnd_imb[1:], gaps[:len(bnd_imb)-1])[0,1] if len(gaps)>0 else 0:+.3f}")
