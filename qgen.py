import sys, numpy as np

NL = 8
N_FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else 'data/synth.raw'
SEED     = int(sys.argv[3]) if len(sys.argv) > 3 else 42
H_LABEL  = int(sys.argv[4]) if len(sys.argv) > 4 else 5

rng = np.random.default_rng(SEED)

TP_RATE       = 0.080
MU_C          = 0.022
LAM_M         = 0.083
DP_RATE       = 0.060
NU_C          = 0.005
LAM_M_DEEP    = 0.030
TP_INSIDE_PROB = 0.06
DP_DIST_MEAN  = 4.0

def init_book():
    aR = np.array([2, 6, 10, 14, 18, 22, 26, 30], dtype=np.int32)
    bR = -aR.copy()
    aN = np.array([3, 4, 4, 4, 3, 3, 2, 2], dtype=np.int32)
    bN = aN.copy()
    aS = aN.copy()
    bS = aN.copy()
    return aR, bR, aS, bS, aN, bN

def compute_rates(aR, bR, aN, bN):
    an0, bn0 = int(aN[0]), int(bN[0])
    and_ = int(aN[1:].sum()); bnd = int(bN[1:].sum())
    tm_a = MU_C * an0 + LAM_M if an0 > 0 else 0
    tm_b = MU_C * bn0 + LAM_M if bn0 > 0 else 0
    dm_a = NU_C * and_ + LAM_M_DEEP if and_ > 0 else 0
    dm_b = NU_C * bnd + LAM_M_DEEP if bnd > 0 else 0
    return np.array([TP_RATE, TP_RATE, tm_a, tm_b, DP_RATE, DP_RATE, dm_a, dm_b])

def apply_tp(aR, bR, aS, bS, aN, bN, side):
    R = bR if side else aR; N = bN if side else aN; S = bS if side else aS
    sp = int(aR[0] - bR[0])
    p_inside = max(0.0, (sp - 2) / max(sp, 1.0))
    inside = (sp > 2) and (rng.random() < p_inside)
    if inside:
        d = 2 * (1 + int(rng.integers(0, max(1, sp // 4))))
        new_R = R[0] - d if side == 0 else R[0] + d
        if side == 0 and new_R <= bR[0]: return
        if side == 1 and new_R >= aR[0]: return
        for k in range(NL - 1, 0, -1):
            R[k] = R[k-1]; N[k] = N[k-1]; S[k] = S[k-1]
        R[0] = new_R; N[0] = 1; S[0] = 1
    else:
        N[0] += 1; S[0] += 1

def apply_tm(aR, bR, aS, bS, aN, bN, side):
    R = bR if side else aR; N = bN if side else aN; S = bS if side else aS
    if N[0] == 0: return
    N[0] -= 1
    if S[0] > 0: S[0] -= 1
    if N[0] == 0:
        for k in range(NL - 1):
            R[k] = R[k+1]; N[k] = N[k+1]; S[k] = S[k+1]
        if N[NL-2] > 0:
            R[NL-1] = R[NL-2] - 2 if side else R[NL-2] + 2
            N[NL-1] = 1; S[NL-1] = 1
        else:
            R[NL-1] = 0; N[NL-1] = 0; S[NL-1] = 0

def apply_dp(aR, bR, aS, bS, aN, bN, side):
    R = bR if side else aR; N = bN if side else aN; S = bS if side else aS
    d = 2 * (int(rng.exponential(DP_DIST_MEAN / 2)) + 1)
    new_R = R[0] - d if side else R[0] + d
    k = 1
    for k in range(1, NL):
        if N[k] == 0: break
        if R[k] == new_R:
            N[k] += 1; S[k] += 1; return
        past = (new_R > R[k]) if side else (new_R < R[k])
        if past: break
    if k >= NL: return
    for j in range(NL - 1, k, -1):
        R[j] = R[j-1]; N[j] = N[j-1]; S[j] = S[j-1]
    R[k] = new_R; N[k] = 1; S[k] = 1

def apply_dm(aR, bR, aS, bS, aN, bN, side):
    R = bR if side else aR; N = bN if side else aN; S = bS if side else aS
    total = int(N[1:].sum())
    if total == 0: return
    u = int(rng.random() * total)
    cum = 0
    pick = 1
    for k in range(1, NL):
        cum += int(N[k])
        if u < cum:
            pick = k; break
    N[pick] -= 1
    if S[pick] > 0: S[pick] -= 1
    if N[pick] == 0:
        for j in range(pick, NL - 1):
            R[j] = R[j+1]; N[j] = N[j+1]; S[j] = S[j+1]
        if N[NL-2] > 0:
            R[NL-1] = R[NL-2] - 2 if side else R[NL-2] + 2
            N[NL-1] = 1; S[NL-1] = 1
        else:
            R[NL-1] = 0; N[NL-1] = 0; S[NL-1] = 0

APPLY = [apply_tp, apply_tp, apply_tm, apply_tm,
         apply_dp, apply_dp, apply_dm, apply_dm]
SIDE  = [0, 1, 0, 1, 0, 1, 0, 1]

def step(aR, bR, aS, bS, aN, bN, dt=1.0):
    t = 0.0
    while t < dt:
        rates = compute_rates(aR, bR, aN, bN)
        total = rates.sum()
        if total <= 0: break
        step_dt = -np.log(max(rng.random(), 1e-12)) / total
        if t + step_dt > dt: break
        t += step_dt
        u = rng.random() * total
        cum = 0.0; pick = 7
        for k in range(8):
            cum += rates[k]
            if u < cum: pick = k; break
        APPLY[pick](aR, bR, aS, bS, aN, bN, SIDE[pick])

def main():
    aR, bR, aS, bS, aN, bN = init_book()
    out = np.zeros((N_FRAMES, 49), dtype=np.int32)
    for i in range(N_FRAMES):
        step(aR, bR, aS, bS, aN, bN)
        out[i, 0:8]   = aR
        out[i, 8:16]  = bR
        out[i, 16:24] = aS
        out[i, 24:32] = bS
        out[i, 32:40] = aN
        out[i, 40:48] = bN
    mid2 = out[:, 0].astype(np.int64) + out[:, 8].astype(np.int64)
    y = np.zeros(N_FRAMES, dtype=np.int32)
    y[:-H_LABEL] = ((mid2[H_LABEL:] - mid2[:-H_LABEL]) * 2).astype(np.int32)
    out[:, 48] = y
    out.tofile(OUT_PATH)
    print(f"wrote {N_FRAMES} frames to {OUT_PATH}")
    print(f"  sp mean={(out[:,0]-out[:,8]).mean():.2f}  std={(out[:,0]-out[:,8]).std():.2f}")
    print(f"  aN[0] mean={out[:,32].mean():.2f}")
    print(f"  y std={y.std()/4:.3f}")

main()
