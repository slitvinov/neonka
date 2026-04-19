"""Replace (or add) sim-dmid feature using 200-replicate average.

For each seed, the sim output has 200 replicate rows after it.  Average
their mid-prices to get a much-lower-variance estimate of E[Δmid | seed]
under our sim's model.
"""
import os, numpy as np, glob

FEAT_DIR = '/tmp/neonka/mlfeat'
SIM_DIR = '/tmp/neonka/sim'
N_REP = int(os.environ.get('N_REP', '200'))
SIM_PREFIX = os.environ.get('SIM_PREFIX', f't55_j{N_REP}')

# Start clean: load the features without any prior sim columns.  We detect
# stale sim columns by re-running ml_feat first to get a pristine base.
# Simpler: just append one column and make sure it replaces any prior append.

mirror = np.load(f'{FEAT_DIR}/mirror.npz')
signs = mirror['signs']
# Base feature count produced by ml_feat.py (before any add_sim_feature.py)
# is 41.  Strip any appended sim columns so re-running is idempotent.
BASE_N = 41
if len(signs) > BASE_N:
    signs = signs[:BASE_N]

# Append mean-sim feature (antisymmetric under side-flip)
new_signs = np.concatenate([signs, [-1.0]])

for s in range(62):
    fp = f'{FEAT_DIR}/s{s}.npz'
    sp = f'{SIM_DIR}/{SIM_PREFIX}_{s}.raw'
    if not os.path.exists(fp) or not os.path.exists(sp): continue
    Z = np.load(fp)
    X = Z['X']
    # Strip stale appended columns beyond BASE_N
    if X.shape[1] > BASE_N:
        X = X[:, :BASE_N]

    r = np.fromfile(sp, dtype=np.int32).reshape(-1, 49)
    n_seeds = len(r) // (N_REP + 1)
    n = min(n_seeds, X.shape[0])
    X = X[:n]
    Z_y = {k: Z[k][:n] for k in ('y1', 'y5', 'y20', 'y55')}

    # Compute per-seed average dmid across 200 replicates
    r_trim = r[:n * (N_REP + 1)].reshape(n, N_REP + 1, 49)
    seed_mid = (r_trim[:, 0, 0] + r_trim[:, 0, 8]) / 2.0
    sim_mids = (r_trim[:, 1:, 0] + r_trim[:, 1:, 8]) / 2.0   # (n, 200)
    dmid_mean = (sim_mids.mean(axis=1) - seed_mid).astype(np.float32)

    X_new = np.hstack([X, dmid_mean[:, None]])
    np.savez(fp, X=X_new, **Z_y)

np.savez(f'{FEAT_DIR}/mirror.npz',
         perm=np.arange(len(new_signs)), signs=new_signs)
print(f'Replaced sim feature with 200-replicate mean.  N_feats = {len(new_signs)}')
