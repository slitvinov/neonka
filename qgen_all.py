import sys, os, numpy as np

NL = 8
NSESS = int(sys.argv[1]) if len(sys.argv) > 1 else 63
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else 'data_synth'
H_LABEL = int(sys.argv[3]) if len(sys.argv) > 3 else 55
RNG_SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 2026
master = np.random.default_rng(RNG_SEED)

os.makedirs(OUT_DIR, exist_ok=True)

def init_book():
    aR = np.array([2, 6, 10, 14, 18, 22, 26, 30], dtype=np.int32)
    bR = -aR.copy()
    aN = np.array([3, 4, 4, 4, 3, 3, 2, 2], dtype=np.int32)
    bN = aN.copy()
    aS = aN.copy()
    bS = aN.copy()
    return aR, bR, aS, bS, aN, bN

def gen_session(params, n_frames, rng):
    aR, bR, aS, bS, aN, bN = init_book()
    out = np.zeros((n_frames, 49), dtype=np.int32)
    TP = params['TP_RATE']; DP = params['DP_RATE']
    MU = params['MU_C']; LM = params['LAM_M']
    NU = params['NU_C']; LD = params['LAM_M_DEEP']
    DM = params['DP_DIST_MEAN']
    TP_SCALE = params.get('TP_SP_SCALE', 0.03)
    for i in range(n_frames):
        t = 0.0
        while t < 1.0:
            an0, bn0 = int(aN[0]), int(bN[0])
            and_ = int(aN[1:].sum()); bnd = int(bN[1:].sum())
            sp_cur = int(aR[0] - bR[0])
            tp_eff = TP + TP_SCALE * max(0, sp_cur - 2)
            tm_a = (MU * an0 + LM) if an0 > 0 else 0.0
            tm_b = (MU * bn0 + LM) if bn0 > 0 else 0.0
            dm_a = (NU * and_ + LD) if and_ > 0 else 0.0
            dm_b = (NU * bnd + LD) if bnd > 0 else 0.0
            rates = np.array([tp_eff, tp_eff, tm_a, tm_b, DP, DP, dm_a, dm_b])
            total = rates.sum()
            if total <= 0: break
            dt = -np.log(max(rng.random(), 1e-12)) / total
            if t + dt > 1.0: break
            t += dt
            u = rng.random() * total
            cum = 0.0; pick = 7
            for k in range(8):
                cum += rates[k]
                if u < cum: pick = k; break
            side = pick & 1
            ev = pick >> 1
            if ev == 0:
                R = bR if side else aR; N = bN if side else aN; S = bS if side else aS
                sp = int(aR[0] - bR[0])
                p_inside = max(0.0, (sp - 2) / max(sp, 1.0))
                if sp > 2 and rng.random() < p_inside:
                    d = 2 * (1 + int(rng.integers(0, max(1, sp // 4))))
                    new_R = R[0] - d if side == 0 else R[0] + d
                    if (side == 0 and new_R > bR[0]) or (side == 1 and new_R < aR[0]):
                        for k in range(NL - 1, 0, -1):
                            R[k] = R[k-1]; N[k] = N[k-1]; S[k] = S[k-1]
                        R[0] = new_R; N[0] = 1; S[0] = 1
                else:
                    N[0] += 1; S[0] += 1
            elif ev == 1:
                R = bR if side else aR; N = bN if side else aN; S = bS if side else aS
                if N[0] == 0: continue
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
            elif ev == 2:
                R = bR if side else aR; N = bN if side else aN; S = bS if side else aS
                d = 2 * (int(rng.exponential(DM / 2)) + 1)
                new_R = R[0] - d if side else R[0] + d
                k = 1
                for k in range(1, NL):
                    if N[k] == 0: break
                    if R[k] == new_R:
                        N[k] += 1; S[k] += 1; break
                    past = (new_R < R[k]) if side else (new_R > R[k])
                    if past: break
                else:
                    continue
                if N[k] == 1 and R[k] != new_R:
                    continue
                if R[k] != new_R and N[k] == 0:
                    for j in range(NL - 1, k, -1):
                        R[j] = R[j-1]; N[j] = N[j-1]; S[j] = S[j-1]
                    R[k] = new_R; N[k] = 1; S[k] = 1
            else:
                R = bR if side else aR; N = bN if side else aN; S = bS if side else aS
                total_n = int(N[1:].sum())
                if total_n == 0: continue
                u2 = int(rng.random() * total_n)
                cum_n = 0; pick_k = 1
                for k in range(1, NL):
                    cum_n += int(N[k])
                    if u2 < cum_n: pick_k = k; break
                N[pick_k] -= 1
                if S[pick_k] > 0: S[pick_k] -= 1
                if N[pick_k] == 0:
                    for j in range(pick_k, NL - 1):
                        R[j] = R[j+1]; N[j] = N[j+1]; S[j] = S[j+1]
                    if N[NL-2] > 0:
                        R[NL-1] = R[NL-2] - 2 if side else R[NL-2] + 2
                        N[NL-1] = 1; S[NL-1] = 1
                    else:
                        R[NL-1] = 0; N[NL-1] = 0; S[NL-1] = 0
        out[i, 0:8] = aR; out[i, 8:16] = bR
        out[i, 16:24] = aS; out[i, 24:32] = bS
        out[i, 32:40] = aN; out[i, 40:48] = bN
    mid2 = out[:, 0].astype(np.int64) + out[:, 8].astype(np.int64)
    y = np.zeros(n_frames, dtype=np.int32)
    y[:-H_LABEL] = ((mid2[H_LABEL:] - mid2[:-H_LABEL]) * 2).astype(np.int32)
    out[:, 48] = y
    return out

all_rows = []
boundaries = [0]
for ses in range(NSESS):
    rng = np.random.default_rng(RNG_SEED * 10000 + ses)
    params = {
        'TP_RATE':       0.06 + 0.04 * master.random(),
        'DP_RATE':       0.04 + 0.04 * master.random(),
        'MU_C':          0.015 + 0.015 * master.random(),
        'LAM_M':         0.05 + 0.07 * master.random(),
        'NU_C':          0.003 + 0.004 * master.random(),
        'LAM_M_DEEP':    0.020 + 0.020 * master.random(),
        'DP_DIST_MEAN':  3.0 + 2.0 * master.random(),
    }
    n_frames = int(100_000 + 200_000 * master.random())
    rows = gen_session(params, n_frames, rng)
    all_rows.append(rows)
    boundaries.append(boundaries[-1] + n_frames)
    sp = (rows[:, 0] - rows[:, 8]).astype(float)
    print(f"  ses {ses:2d}  n={n_frames:>6}  sp={sp.mean():5.2f}  aN0={rows[:,32].mean():.2f}  "
          f"y_std={rows[:,48].std()/4:5.2f}  params μ_c={params['MU_C']:.3f} λ_m={params['LAM_M']:.3f}")

raw = np.concatenate(all_rows, axis=0)
raw.tofile(f'{OUT_DIR}/train.raw')
np.array(boundaries, dtype=np.int64).tofile(f'{OUT_DIR}/sessions.raw')
print(f"\nwrote {raw.shape[0]} total rows, {NSESS} sessions, to {OUT_DIR}/")
