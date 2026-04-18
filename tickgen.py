"""tickgen.py — tick-quantized LOB simulator (prototype).

Same book dynamics as qgen.py (same apply_tp/tm/dp/dm, same rate formulas)
but each frame is one categorical draw over {no-event, 8 event types} — never
fires 2+ events per tick. Matches real data's 89%-1-event signature.

qgen step():  Gillespie inner loop → Poisson count per frame
              → classify gives 42/37/15/4 % (0/1/2/3+ events)

tickgen step(): one draw per tick
              → classify gives ~10/90/0/0 %    (0/1/2/3+ events)

Usage: python3 tickgen.py [n_frames] [seed]
"""
import sys, numpy as np

N_FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
SEED     = int(sys.argv[2]) if len(sys.argv) > 2 else 42
P_NOEV   = 0.10
rng = np.random.default_rng(SEED)

TP_RATE, MU_C, LAM_M = 0.080, 0.022, 0.083
DP_RATE, NU_C, LAM_MD = 0.060, 0.005, 0.030
NL = 8

def init_book():
    aR = np.array([2, 6, 10, 14, 18, 22, 26, 30], dtype=np.int32)
    bR = -aR.copy()
    aN = np.array([3, 4, 4, 4, 3, 3, 2, 2], dtype=np.int32)
    return aR, bR, aN.copy(), aN.copy(), aN.copy(), aN.copy()

def compute_rates(aN, bN):
    an0, bn0 = int(aN[0]), int(bN[0])
    and_, bnd = int(aN[1:].sum()), int(bN[1:].sum())
    tm_a = MU_C * an0 + LAM_M  if an0 > 0 else 0
    tm_b = MU_C * bn0 + LAM_M  if bn0 > 0 else 0
    dm_a = NU_C * and_ + LAM_MD if and_ > 0 else 0
    dm_b = NU_C * bnd + LAM_MD if bnd > 0 else 0
    return np.array([TP_RATE, TP_RATE, tm_a, tm_b, DP_RATE, DP_RATE, dm_a, dm_b])

def apply_tp(aR, bR, aS, bS, aN, bN, side):
    R = bR if side else aR; N = bN if side else aN; S = bS if side else aS
    sp = int(aR[0] - bR[0])
    p_in = max(0.0, (sp - 2) / max(sp, 1.0))
    if sp > 2 and rng.random() < p_in:
        d = 2 * (1 + int(rng.integers(0, max(1, sp // 4))))
        newR = R[0] - d if side == 0 else R[0] + d
        if side == 0 and newR <= bR[0]: return
        if side == 1 and newR >= aR[0]: return
        for k in range(NL-1, 0, -1): R[k], N[k], S[k] = R[k-1], N[k-1], S[k-1]
        R[0], N[0], S[0] = newR, 1, 1
    else:
        N[0] += 1; S[0] += 1

def apply_tm(aR, bR, aS, bS, aN, bN, side):
    R = bR if side else aR; N = bN if side else aN; S = bS if side else aS
    if N[0] == 0: return
    N[0] -= 1
    if S[0] > 0: S[0] -= 1
    if N[0] == 0:
        for k in range(NL-1): R[k], N[k], S[k] = R[k+1], N[k+1], S[k+1]
        if N[NL-2] > 0:
            R[NL-1] = R[NL-2] - 2 if side else R[NL-2] + 2
            N[NL-1], S[NL-1] = 1, 1
        else:
            R[NL-1], N[NL-1], S[NL-1] = 0, 0, 0

def apply_dp(aR, bR, aS, bS, aN, bN, side):
    R = bR if side else aR; N = bN if side else aN; S = bS if side else aS
    d = 2 * (int(rng.exponential(2.0)) + 1)
    newR = R[0] - d if side else R[0] + d
    k = 1
    for k in range(1, NL):
        if N[k] == 0: break
        if R[k] == newR:
            N[k] += 1; S[k] += 1; return
        past = (newR > R[k]) if side else (newR < R[k])
        if past: break
    if k >= NL: return
    for j in range(NL-1, k, -1): R[j], N[j], S[j] = R[j-1], N[j-1], S[j-1]
    R[k], N[k], S[k] = newR, 1, 1

def apply_dm(aR, bR, aS, bS, aN, bN, side):
    R = bR if side else aR; N = bN if side else aN; S = bS if side else aS
    total = int(N[1:].sum())
    if total == 0: return
    u = int(rng.random() * total)
    cum, pick = 0, 1
    for k in range(1, NL):
        cum += int(N[k])
        if u < cum: pick = k; break
    N[pick] -= 1
    if S[pick] > 0: S[pick] -= 1
    if N[pick] == 0:
        for j in range(pick, NL-1): R[j], N[j], S[j] = R[j+1], N[j+1], S[j+1]
        if N[NL-2] > 0:
            R[NL-1] = R[NL-2] - 2 if side else R[NL-2] + 2
            N[NL-1], S[NL-1] = 1, 1
        else:
            R[NL-1], N[NL-1], S[NL-1] = 0, 0, 0

APPLY = [apply_tp, apply_tp, apply_tm, apply_tm,
         apply_dp, apply_dp, apply_dm, apply_dm]
SIDE  = [0, 1, 0, 1, 0, 1, 0, 1]

def step_tick(aR, bR, aS, bS, aN, bN):
    rates = compute_rates(aN, bN)
    tot = rates.sum()
    if tot <= 0: return
    probs = np.concatenate([[P_NOEV], rates * (1 - P_NOEV) / tot])
    k = rng.choice(9, p=probs)
    if k > 0:
        APPLY[k-1](aR, bR, aS, bS, aN, bN, SIDE[k-1])

def main():
    aR, bR, aS, bS, aN, bN = init_book()
    out = np.zeros((N_FRAMES, 49), dtype=np.int32)
    for i in range(N_FRAMES):
        step_tick(aR, bR, aS, bS, aN, bN)
        out[i, 0:8]   = aR; out[i, 8:16]  = bR
        out[i, 16:24] = aS; out[i, 24:32] = bS
        out[i, 32:40] = aN; out[i, 40:48] = bN
    out.tofile('/tmp/tickgen.raw')

    # self-check: count events per pair
    def walk(R0, N0, R1, N1, diff):
        i = j = n = 0
        while i < 8 and j < 8 and N0[i] != 0 and N1[j] != 0:
            d = diff * (R1[j] - R0[i])
            if d < 0: n += int(N1[j]); j += 1
            elif d == 0:
                dn = int(N1[j]) - int(N0[i])
                if dn: n += abs(dn)
                i += 1; j += 1
            else: n += int(N0[i]); i += 1
        return n
    ev = np.empty(N_FRAMES - 1, dtype=np.int32)
    for k in range(N_FRAMES - 1):
        p, c = out[k], out[k+1]
        ev[k] = walk(p[0:8], p[32:40], c[0:8], c[32:40], +1) + \
                walk(p[8:16], p[40:48], c[8:16], c[40:48], -1)
    u, cnt = np.unique(ev, return_counts=True)
    print(f"tickgen  ({N_FRAMES} frames, mean events/pair = {ev.mean():.3f})")
    for uu, cc in zip(u[:8], cnt[:8]):
        print(f"  {uu} events: {100*cc/len(ev):5.1f}%")

main()
