"""Add sim-based T=55 mid-price prediction as a feature, retrain ML.

For each seed row in ml_feat's output, the sim has already been run to t+T.
We extract dmid_sim = sim_mid(t+T) − seed_mid(t) and inject it into the
feature vector.  Then retrain LOSO and measure R².

The sim feature is "symmetric under side-flip? NO" — it's a directional
prediction (antisymmetric).  Under the side-flip, dmid → −dmid.
"""
import os, numpy as np

FEAT_DIR = '/tmp/neonka/mlfeat'
SIM_DIR = '/tmp/neonka/sim'

# Original mirror vector — we'll append one new sign.
mirror = np.load(f'{FEAT_DIR}/mirror.npz')
signs = mirror['signs']
new_signs = np.concatenate([signs, [-1.0]])   # dmid_sim is antisymmetric

STRIDE = 100
for s in range(62):
    fp = f'{FEAT_DIR}/s{s}.npz'
    sp = f'{SIM_DIR}/t55_h8_{s}.raw'
    if not os.path.exists(fp) or not os.path.exists(sp): continue
    Z = np.load(fp)
    X, y1, y5, y20, y55 = Z['X'], Z['y1'], Z['y5'], Z['y20'], Z['y55']
    r = np.fromfile(sp, dtype=np.int32).reshape(-1, 49)
    # Sim pairs are at stride K=100 over the session's IDLE rows.
    # Same stride as ml_feat, so n_pairs should match n_seeds.
    n_seeds = X.shape[0]
    n_pairs = len(r) // 2
    if n_pairs < n_seeds:
        # truncate features to what the sim has
        n = n_pairs
        X = X[:n]; y1 = y1[:n]; y5 = y5[:n]; y20 = y20[:n]; y55 = y55[:n]
    else:
        n = n_seeds
    seed_mid = (r[:2*n:2, 0] + r[:2*n:2, 8]) / 2.0
    sim_mid  = (r[1:2*n+1:2, 0] + r[1:2*n+1:2, 8]) / 2.0
    dmid_sim = (sim_mid - seed_mid).astype(np.float32)
    # Append as new column
    X_new = np.hstack([X, dmid_sim[:, None]])
    np.savez(fp, X=X_new, y1=y1, y5=y5, y20=y20, y55=y55)

np.savez(f'{FEAT_DIR}/mirror.npz',
         perm=np.arange(len(new_signs)), signs=new_signs)
print(f'Added dmid_sim feature.  N_feats = {len(new_signs)}')
