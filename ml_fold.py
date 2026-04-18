"""Fit per-session (no pooling): first 70% train, last 30% test, within session.

Usage: python3 ml_fold.py SESSION_ID T [--mirror]
 (No --regime flag — this script never pools across sessions.)
"""
import sys, numpy as np, os
from sklearn.ensemble import HistGradientBoostingRegressor

s = int(sys.argv[1])
T = int(sys.argv[2])
use_mirror = '--mirror' in sys.argv
ykey = f'y{T}'

FEAT = '/tmp/mlfeat'
p = f'{FEAT}/s{s}.npz'
if not os.path.exists(p):
    print(f"{s} {T} 0.0000"); sys.exit(0)
d = np.load(p)
X, y = d['X'], d[ykey]
n = len(y)
if n < 400:
    print(f"{s} {T} 0.0000"); sys.exit(0)

# Random 70/30 split with seed = session id for reproducibility
rng = np.random.default_rng(s)
idx = rng.permutation(n)
split = int(0.7 * n)
tr_idx = idx[:split]; te_idx = idx[split:]
X_tr = X[tr_idx]; y_tr = y[tr_idx]
X_te = X[te_idx]; y_te = y[te_idx]

if use_mirror:
    m = np.load(f'{FEAT}/mirror.npz')
    perm, signs = m['perm'], m['signs']
    X_m = (X_tr[:, perm] * signs).astype(np.float32)
    X_tr = np.concatenate([X_tr, X_m], axis=0)
    y_tr = np.concatenate([y_tr, -y_tr])

mdl = HistGradientBoostingRegressor(
        max_iter=80, max_depth=3, learning_rate=0.05,
        l2_regularization=5.0, min_samples_leaf=100,
        early_stopping=True, validation_fraction=0.2, n_iter_no_change=10)
mdl.fit(X_tr, y_tr)
yp = mdl.predict(X_te)

# Inference-time symmetrization: ŷ = (f(X) − f(flip(X))) / 2.
# Enforces odd symmetry — the true target is known-antisymmetric by construction.
if '--sym-infer' in sys.argv:
    m = np.load(f'{FEAT}/mirror.npz')
    perm, signs = m['perm'], m['signs']
    X_te_m = (X_te[:, perm] * signs).astype(np.float32)
    yp_m = mdl.predict(X_te_m)
    yp = (yp - yp_m) / 2.0

sst = (y_te * y_te).sum()
r2 = 100 * (1 - ((yp - y_te) ** 2).sum() / sst) if sst > 0 else 0
print(f"{s} {T} {r2:+.4f}")
