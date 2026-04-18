import sys, os, glob
import numpy as np

TABLES = "data/tables"
EVENTS = ["tp", "tm", "dp", "dm"]
SIDES  = ["a", "b"]
IMB_BINS = [0, 1, 2]
SPS = range(2, 21, 2)

def load(ses, event, side, imb):
    p = f"{TABLES}/{ses:02d}/{event}.{side}.imb{imb}.rates"
    if not os.path.exists(p): return None
    data = np.loadtxt(p)
    if data.ndim == 1: data = data.reshape(1, -1)
    if len(data) == 0: return None
    return dict(zip(data[:, 0].astype(int), data[:, 1]))

def sessions_avail():
    return sorted(int(os.path.basename(d)) for d in glob.glob(f"{TABLES}/[0-9][0-9]"))

def n_rate(ses, imb):
    p = f"{TABLES}/{ses:02d}/n.imb{imb}.rates"
    if not os.path.exists(p): return None
    d = np.loadtxt(p)
    if d.ndim == 1: d = d.reshape(1, -1)
    if len(d) == 0: return None
    return dict(zip(d[:, 0].astype(int), d[:, 1]))

def matrix_at(event, side, imb, sp_targets):
    sess = sessions_avail()
    M = np.full((len(sess), len(sp_targets)), np.nan)
    for i, s in enumerate(sess):
        t = load(s, event, side, imb)
        if t is None: continue
        for j, sp in enumerate(sp_targets):
            if sp in t: M[i, j] = t[sp]
    return sess, M

def cmd_summary():
    sess = sessions_avail()
    print(f"{len(sess)} sessions available")
    print(f"\n{'ses':>4}  {'sp_mean':>7}  " + "  ".join(f"nr_im{b}" for b in IMB_BINS) +
          f"  {'act_rate':>9}  {'tp_a(6)':>7}  {'tm_a(6)':>7}  {'dm_a(6)':>7}")
    import numpy as np
    r = np.fromfile('data/train.raw', dtype=np.int32).reshape(-1, 49)
    b = np.fromfile('data/sessions.raw', dtype=np.int64)
    for s in sess:
        lo, hi = int(b[s]), int(b[s+1])
        sp_mn = (r[lo:hi, 0] - r[lo:hi, 8]).astype(float).mean()
        nrs = []
        for im in IMB_BINS:
            t = n_rate(s, im)
            if t is None or 6 not in t: nrs.append(np.nan); continue
            nrs.append(t[6])
        act = 1 - np.nanmean(nrs)
        tp_a_6 = (load(s, "tp", "a", 1) or {}).get(6, np.nan)
        tm_a_6 = (load(s, "tm", "a", 1) or {}).get(6, np.nan)
        dm_a_6 = (load(s, "dm", "a", 1) or {}).get(6, np.nan)
        nr_str = "  ".join(f"{x:6.3f}" if not np.isnan(x) else "   n/a" for x in nrs)
        print(f"{s:>4}  {sp_mn:>7.2f}  {nr_str}  {act:>9.4f}  "
              f"{tp_a_6:>7.4f}  {tm_a_6:>7.4f}  {dm_a_6:>7.4f}")

def cmd_corr(event, side, imb):
    sp_targets = list(SPS)
    sess, M = matrix_at(event, side, imb, sp_targets)
    print(f"\n{event}.{side}.imb{imb} — rate values across sessions, at common spreads")
    print(f"  sp:  " + "  ".join(f"{s:>7}" for s in sp_targets))
    print(f"  " + "-" * (8 + 9 * len(sp_targets)))
    good = ~np.isnan(M).any(axis=1)
    Mg = M[good]
    sess_g = [sess[i] for i, g in enumerate(good) if g]
    if len(Mg) < 3:
        print("  insufficient data")
        return
    # Per-sp: mean, std, CV
    print(f"  mean: " + "  ".join(f"{x:>7.4f}" for x in Mg.mean(0)))
    print(f"   std: " + "  ".join(f"{x:>7.4f}" for x in Mg.std(0)))
    print(f"  CV%:  " + "  ".join(f"{100*s/m:>7.1f}" if m > 0 else "   n/a"
                                   for m, s in zip(Mg.mean(0), Mg.std(0))))
    # Cross-session correlation of the rate-vs-sp curve
    # Correlation between two rows
    C = np.corrcoef(Mg)
    print(f"\n  pairwise rate-curve correlations (n_sess={len(Mg)}):")
    print(f"  median r = {np.median(C[np.triu_indices(len(Mg), 1)]):.3f}")
    print(f"  min  r   = {C[np.triu_indices(len(Mg), 1)].min():.3f}")
    print(f"  p10  r   = {np.percentile(C[np.triu_indices(len(Mg), 1)], 10):.3f}")

def cmd_pool(event, side, imb, k_clusters=3):
    sp_targets = list(SPS)
    sess, M = matrix_at(event, side, imb, sp_targets)
    good = ~np.isnan(M).any(axis=1)
    Mg = M[good]
    sess_g = [sess[i] for i, g in enumerate(good) if g]
    if len(Mg) < k_clusters + 1:
        print("insufficient data for clustering")
        return
    # Normalize each session's curve to unit L2 to compare SHAPES
    shapes = Mg / (np.linalg.norm(Mg, axis=1, keepdims=True) + 1e-12)
    # simple k-means on shapes
    rng = np.random.default_rng(0)
    idx = rng.choice(len(shapes), k_clusters, replace=False)
    cents = shapes[idx].copy()
    for _ in range(50):
        d = np.linalg.norm(shapes[:, None, :] - cents[None, :, :], axis=2)
        assign = d.argmin(axis=1)
        new = np.zeros_like(cents)
        for c in range(k_clusters):
            m = assign == c
            if m.sum() > 0: new[c] = shapes[m].mean(0)
            else: new[c] = cents[c]
        if np.allclose(cents, new): break
        cents = new
    print(f"\n{event}.{side}.imb{imb} — shape-clustering into {k_clusters} groups")
    for c in range(k_clusters):
        m = assign == c
        members = [sess_g[i] for i in range(len(sess_g)) if m[i]]
        # also show cluster centroid rates (unnormalized: mean of Mg in cluster)
        if m.sum() > 0:
            cent_rates = Mg[m].mean(0)
            print(f"\n  cluster {c}  ({m.sum()} sessions):")
            print(f"    members: {members}")
            print(f"    mean rate by sp: " + "  ".join(f"sp{sp}={r:.4f}" for sp, r in zip(sp_targets, cent_rates)))

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "summary"
    if cmd == "summary":
        cmd_summary()
    elif cmd == "corr":
        event = sys.argv[2] if len(sys.argv) > 2 else "tp"
        side  = sys.argv[3] if len(sys.argv) > 3 else "a"
        imb   = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        cmd_corr(event, side, imb)
    elif cmd == "pool":
        event = sys.argv[2] if len(sys.argv) > 2 else "tp"
        side  = sys.argv[3] if len(sys.argv) > 3 else "a"
        imb   = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        k     = int(sys.argv[5]) if len(sys.argv) > 5 else 3
        cmd_pool(event, side, imb, k)
    elif cmd == "all-corr":
        for ev in EVENTS:
            for sd in SIDES:
                for im in IMB_BINS:
                    cmd_corr(ev, sd, im)
    else:
        print("usage: ratediff.py {summary|corr EVENT SIDE IMB|pool EVENT SIDE IMB [K]|all-corr}",
              file=sys.stderr)
        sys.exit(1)
