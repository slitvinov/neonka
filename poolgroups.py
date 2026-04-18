import os, sys, glob
import numpy as np

TABLES = "data/tables"
K = 3

def list_sessions():
    return sorted(int(os.path.basename(d)) for d in glob.glob(f"{TABLES}/[0-9][0-9]"))

def load_sp_imb_raw(ses):
    p = f"{TABLES}/{ses:02d}/sp_imb_rates.raw"
    if not os.path.exists(p): return None
    a = np.loadtxt(p)
    if a.ndim == 1: a = a.reshape(1, -1)
    return a

def rate_matrix(sess, sp_target=6):
    EVTS = [(e, s, i) for e in ["tp","tm","dp","dm"] for s in ["a","b"] for i in [0,1,2]]
    X = np.full((len(sess), len(EVTS)), np.nan)
    for k, ses in enumerate(sess):
        data = load_sp_imb_raw(ses)
        if data is None: continue
        for row in data:
            sp, imb = int(row[0]), int(row[1])
            if sp != sp_target: continue
            ntics = row[2]
            if ntics <= 0: continue
            a_vals = row[4:9]; b_vals = row[9:14]
            vals = {(("tp","tm","dp","dm","r")[k], "a", imb): a_vals[k]/ntics for k in range(4)}
            vals.update({(("tp","tm","dp","dm","r")[k], "b", imb): b_vals[k]/ntics for k in range(4)})
            for j, (e, sd, im) in enumerate(EVTS):
                if (e, sd, im) in vals:
                    X[k, j] = vals[(e, sd, im)]
    return X

def assign_groups(sess, K=3):
    X = rate_matrix(sess)
    mask = ~np.isnan(X).any(axis=1)
    Xg = X[mask]; sess_g = [sess[i] for i in range(len(sess)) if mask[i]]
    mu = Xg.mean(0); Xc = (Xg - mu) / (Xg.std(0) + 1e-12)
    U, Sv, Vt = np.linalg.svd(Xc, full_matrices=False)
    pc1 = Xc @ Vt[0]
    qs = np.quantile(pc1, np.linspace(0, 1, K+1)[1:-1])
    groups = np.zeros(len(sess), dtype=int)
    for i, s in enumerate(sess):
        if not mask[i]:
            groups[i] = -1
            continue
        idx = sess_g.index(s)
        g = int(np.searchsorted(qs, pc1[idx]))
        groups[i] = g
    pc1_full = np.full(len(sess), np.nan)
    for i, s in enumerate(sess):
        if mask[i]:
            pc1_full[i] = pc1[sess_g.index(s)]
    return groups, pc1_full

def pool_group(members):
    acc = {}
    for ses in members:
        data = load_sp_imb_raw(ses)
        if data is None: continue
        for row in data:
            sp, imb = int(row[0]), int(row[1])
            key = (sp, imb)
            if key not in acc:
                acc[key] = np.zeros(12)
            acc[key] += row[2:]
    return acc

NAMES = ["tp", "tm", "dp", "dm", "r"]

def write_pooled_to_session(ses, pooled):
    d = f"{TABLES}/{ses:02d}"
    os.makedirs(d, exist_ok=True)
    imbs = sorted({k[1] for k in pooled})
    by_imb_nt = {im: {} for im in imbs}  # imb -> sp -> ntics
    by_imb_n  = {im: {} for im in imbs}  # imb -> sp -> n (nothing)
    by_imb_e  = {im: {} for im in imbs}  # imb -> sp -> 10-col event vector
    for (sp, imb), v in pooled.items():
        by_imb_nt[imb][sp] = v[0]
        by_imb_n[imb][sp]  = v[1]
        by_imb_e[imb][sp]  = v[2:]
    for imb in imbs:
        with open(f"{d}/n.imb{imb}.rates", "w") as f:
            for sp in sorted(by_imb_n[imb]):
                nt = by_imb_nt[imb][sp]
                if nt > 0:
                    f.write(f"{sp} {by_imb_n[imb][sp]/nt:.10f}\n")
        for ei, ev in enumerate(NAMES):
            for si, sd in enumerate(("a","b")):
                col = ei + si * 5
                with open(f"{d}/{ev}.{sd}.imb{imb}.rates", "w") as f:
                    for sp in sorted(by_imb_e[imb]):
                        nt = by_imb_nt[imb][sp]
                        if nt > 0:
                            f.write(f"{sp} {by_imb_e[imb][sp][col]/nt:.10f}\n")

def main():
    sess = list_sessions()
    print(f"found {len(sess)} sessions")
    groups, pc1 = assign_groups(sess, K=K)
    print(f"\nPC1-based groups (K={K}):")
    for g in range(K):
        members = [sess[i] for i in range(len(sess)) if groups[i] == g]
        pc_vals = [pc1[i] for i in range(len(sess)) if groups[i] == g]
        print(f"  group {g}  n={len(members):2d}  PC1 in [{min(pc_vals):+.2f}, {max(pc_vals):+.2f}]")
        print(f"    members: {members}")
    unassigned = [sess[i] for i in range(len(sess)) if groups[i] == -1]
    if unassigned:
        print(f"  unassigned (missing data): {unassigned}")
    # Save backup + write pooled
    for g in range(K):
        members = [sess[i] for i in range(len(sess)) if groups[i] == g]
        if not members: continue
        pooled = pool_group(members)
        print(f"\n  group {g}: pooled {len(members)} sessions, {len(pooled)} (sp,imb) buckets")
        for ses in members:
            # backup existing per-session rate files to *.persess
            d = f"{TABLES}/{ses:02d}"
            for f in glob.glob(f"{d}/*.imb*.rates"):
                bak = f.replace(".rates", ".persess")
                if not os.path.exists(bak):
                    os.rename(f, bak)
            write_pooled_to_session(ses, pooled)
    print("\nDone. per-session rate files backed up with .persess suffix.")
    print("Run avgsess.sh to test with pooled tables.")

if __name__ == "__main__":
    main()
