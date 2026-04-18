"""Fit power-law tail for each session's refill distribution.
Writes /tmp/tables{s}/refill.{a,b}.tail with: alpha k_cutoff f_tail max_obs.
  alpha    : power-law exponent (α < −1 for integrable tail)
  k_cutoff : smallest k where parametric tail takes over (= max observed k + 2)
  f_tail   : total tail mass beyond k_cutoff (prob of drawing from parametric)
"""
import os, numpy as np

ROOT = '/tmp'

def load_refill(path):
    if not os.path.exists(path): return np.array([]), np.array([])
    arr = np.loadtxt(path, ndmin=2)
    if arr.size == 0: return np.array([]), np.array([])
    return arr[:, 0].astype(int), arr[:, 1].astype(np.int64)

def fit_power_alpha(k, p, k_min=4):
    mask = (k >= k_min) & (p > 0)
    if mask.sum() < 3: return None, None
    logk = np.log(k[mask]); logp = np.log(p[mask])
    beta, alpha = np.polyfit(logk, logp, 1)
    return beta, alpha     # p ≈ exp(alpha) · k^beta

def write_tail_params(out_path, k_arr, c_arr):
    """Fit tail to one side (ask or bid) and save."""
    if len(k_arr) == 0:
        with open(out_path, 'w') as f:
            f.write("# no refill data\n0 0 0 0\n")
        return
    N = c_arr.sum()
    p = c_arr / N
    max_obs = int(k_arr.max())
    alpha_pow, C_pow = fit_power_alpha(k_arr, p, k_min=4)
    if alpha_pow is None or alpha_pow >= -1.05:
        # Not enough tail or tail too heavy — skip parametric
        with open(out_path, 'w') as f:
            f.write("# tail fit unreliable; hybrid disabled\n")
            f.write("alpha 0.0\n")
            f.write("k_cutoff %d\n" % (max_obs + 2))
            f.write("f_tail 0.0\n")
            f.write("max_obs %d\n" % max_obs)
        return
    # Integrate extrapolated tail mass beyond max_obs (continuous approx).
    # ∫_{k_cutoff}^{∞} exp(C) · k^α dk = exp(C) · k_cutoff^(α+1) / (−α−1).
    # This gives the ADDITIONAL probability beyond what's seen. Use it as f_tail
    # (the sampler triggers parametric branch with this prob).
    k_cutoff = max_obs + 2
    extrap_mass = np.exp(C_pow) * (k_cutoff ** (alpha_pow + 1)) / (-alpha_pow - 1)
    # Clip to sanity: cumulative prob in [0, 0.1]
    f_tail = float(min(max(extrap_mass, 0.0), 0.1))
    with open(out_path, 'w') as f:
        f.write(f"alpha {alpha_pow:.6f}\n")
        f.write(f"k_cutoff {k_cutoff}\n")
        f.write(f"f_tail {f_tail:.6e}\n")
        f.write(f"max_obs {max_obs}\n")

for sid in range(62):
    d = f'{ROOT}/tables{sid}'
    if not os.path.isdir(d): continue
    for side in ['a', 'b']:
        k, c = load_refill(f'{d}/refill.{side}.own')
        write_tail_params(f'{d}/refill.{side}.tail', k, c)
    # Echo one sample
    if sid in (0, 45, 53, 61):
        for side in ['a', 'b']:
            with open(f'{d}/refill.{side}.tail') as f:
                lines = [l.strip() for l in f if not l.startswith('#')]
            print(f"ses{sid:2d} {side}:  {' '.join(lines)}")
print("done — wrote refill.{a,b}.tail to all 62 sessions")
