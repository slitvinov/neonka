"""Per-session features with even/odd (symmetric/antisymmetric) decomposition.

Under side-flip:
  symmetric features  (S_sum, sp, imb0², |imb0|, ...): unchanged
  antisymmetric       (imb0, N_diff, OFI, momentum, ...): negate

Layout (no bid/ask swap needed — already decomposed):
  sym            → sign +1
  antisym        → sign -1
Mirror metadata is now just a sign vector; permutation is identity.
"""
import numpy as np, os, time

RECSZ = 54 * 4
STRIDE = 100
T_LIST = [1, 5, 20, 55]
W_OFI_LIST = [50, 100, 500, 2000]
W_MOM_LIST = [50, 200]
REGIME_SPLIT = 52

SIGN_TOP_CNT = np.array([-1, +1, +1, -1, 0, 0, 0, 0])
SIGN_DEEP_CNT = np.array([0, 0, 0, 0, -1, +1, +1, -1])

OUTDIR = '/tmp/neonka/mlfeat'
os.makedirs(OUTDIR, exist_ok=True)

def cks_row_ofi(aR0, aS0, bR0, bS0):
    n = len(aR0)
    e_bid = np.zeros(n - 1, dtype=np.float64)
    e_ask = np.zeros(n - 1, dtype=np.float64)
    dPb = bR0[1:] - bR0[:-1]
    dPa = aR0[1:] - aR0[:-1]
    up = dPb > 0; flat = dPb == 0; down = dPb < 0
    e_bid[up]   = bS0[1:][up]
    e_bid[flat] = bS0[1:][flat] - bS0[:-1][flat]
    e_bid[down] = -bS0[:-1][down]
    up_a = dPa > 0; flat_a = dPa == 0; down_a = dPa < 0
    e_ask[up_a]   = aS0[:-1][up_a]
    e_ask[flat_a] = aS0[:-1][flat_a] - aS0[1:][flat_a]
    e_ask[down_a] = -aS0[1:][down_a]
    return e_bid - e_ask

def load_session(s, offs, ev_mm):
    lo_r = int(offs[s]) // RECSZ; hi_r = int(offs[s+1]) // RECSZ
    block = ev_mm[lo_r:hi_r]
    types = block[:, 0]
    is_idle = types == 8
    idles = block[is_idle]
    books = idles[:, 5:54].astype(np.float64)
    idle_t = idles[:, 1].astype(np.int64)
    n = len(books)
    ev_t  = block[~is_idle, 1].astype(np.int64)
    ev_ty = types[~is_idle].astype(np.int64)
    # Pre-event aN[0] / bN[0] for tm-split (N>1 → tm_q, N=1 → tm_c)
    ev_aN0 = block[~is_idle, 37].astype(np.int64)
    ev_bN0 = block[~is_idle, 45].astype(np.int64)
    # Pooled 6-D type id per event: tp=0, tm_q=1, tm_c=2, dp=3, dm=4, hp=5
    ev_pt = np.zeros(len(ev_ty), dtype=np.int64)
    ev_pt[(ev_ty == 0) | (ev_ty == 1)] = 0                        # tp
    is_tm_a = (ev_ty == 2); is_tm_b = (ev_ty == 3)
    ev_pt[is_tm_a] = np.where(ev_aN0[is_tm_a] > 1, 1, 2)
    ev_pt[is_tm_b] = np.where(ev_bN0[is_tm_b] > 1, 1, 2)
    ev_pt[(ev_ty == 4) | (ev_ty == 5)] = 3                        # dp
    ev_pt[(ev_ty == 6) | (ev_ty == 7)] = 4                        # dm
    ev_pt[ev_ty >= 9] = 5                                         # hp (if present)

    cum_top_cnt = np.concatenate([[0], np.cumsum(SIGN_TOP_CNT[ev_ty])])
    cum_deep_cnt = np.concatenate([[0], np.cumsum(SIGN_DEEP_CNT[ev_ty])])

    aR0_all = books[:, 0]; bR0_all = books[:, 8]
    aS0_all = books[:, 16]; bS0_all = books[:, 24]
    ofi_row = cks_row_ofi(aR0_all, aS0_all, bR0_all, bS0_all)
    cum_ofi = np.concatenate([[0.0], np.cumsum(ofi_row)])

    T_max = max(T_LIST); W_mom_max = max(W_MOM_LIST)
    start = W_mom_max
    seeds = np.arange(start, n - T_max, STRIDE)
    if len(seeds) == 0:
        return None
    seed_row = idle_t[seeds]
    end_idx = np.searchsorted(ev_t, seed_row, side='right')

    Xb = books[seeds]
    aR = Xb[:, 0:8];  bR = Xb[:, 8:16]
    aS = Xb[:, 16:24]; bS = Xb[:, 24:32]
    aN = Xb[:, 32:40]; bN = Xb[:, 40:48]
    mid = (aR[:, 0] + bR[:, 0]) / 2.0
    sp = aR[:, 0] - bR[:, 0]
    imb0 = (aN[:, 0] - bN[:, 0]) / np.maximum(aN[:, 0] + bN[:, 0], 1)
    imb1 = (aN[:, 1] - bN[:, 1]) / np.maximum(aN[:, 1] + bN[:, 1], 1)
    ida = aN[:, 1:].sum(axis=1); idb = bN[:, 1:].sum(axis=1)
    imb_deep = (ida - idb) / np.maximum(ida + idb, 1)
    imb0_S = (aS[:, 0] - bS[:, 0]) / np.maximum(aS[:, 0] + bS[:, 0], 1)

    # Even/odd decomposition of per-side features
    N_sum  = (aN[:, :3] + bN[:, :3])          # (n, 3)  sym
    N_diff = (aN[:, :3] - bN[:, :3])          # (n, 3)  antisym
    S_sum  = (aS[:, :1] + bS[:, :1])          # (n, 1)  sym
    S_diff = (aS[:, :1] - bS[:, :1])          # (n, 1)  antisym
    aR_gap = aR[:, 1:] - aR[:, :-1]
    bR_gap = bR[:, :-1] - bR[:, 1:]
    gap_sum  = (aR_gap[:, :3] + bR_gap[:, :3])   # (n, 3)  sym
    gap_diff = (aR_gap[:, :3] - bR_gap[:, :3])   # (n, 3)  antisym
    top_queue = aN[:, 0] + bN[:, 0]              # sym

    # Symmetric nonlinearities
    imb0_sq     = imb0 ** 2
    abs_imb0    = np.abs(imb0)
    sp_abs_imb0 = sp * abs_imb0
    aN0_bN0     = aN[:, 0] * bN[:, 0]

    # Antisymmetric × symmetric product
    imb0_totalq = imb0 * top_queue

    base_sym = [
        sp[:, None],                             # 0  sym
        N_sum,                                   # 1,2,3  sym
        S_sum,                                   # 4  sym
        gap_sum,                                 # 5,6,7  sym
        top_queue[:, None],                      # 8  sym
        imb0_sq[:, None],                        # 9  sym
        abs_imb0[:, None],                       # 10 sym
        sp_abs_imb0[:, None],                    # 11 sym
        aN0_bN0[:, None],                        # 12 sym
    ]
    base_anti = [
        imb0[:, None], imb1[:, None],            # 13,14 antisym
        imb_deep[:, None], imb0_S[:, None],      # 15,16 antisym
        N_diff,                                  # 17,18,19 antisym
        S_diff,                                  # 20 antisym
        gap_diff,                                # 21,22,23 antisym
        imb0_totalq[:, None],                    # 24 antisym
    ]
    n_sym_base = 13
    n_anti_base = 12

    # Hawkes memory φ per POOLED event type (6-D: tp, tm_q, tm_c, dp, dm, hp).
    # Single-exponential kernel: φ_c(t) = Σ_{past events of c} exp(−β (t − t_ev)).
    # Intensity: λ_c(t) = μ_c + Σ_j α_{c,j} φ_j(t).
    pp = f'/tmp/neonka/hawkes/{s}.params'
    D = 6
    h_beta = 0.05
    h_mu = np.zeros(D); h_alpha = np.zeros((D, D))
    if os.path.exists(pp):
        with open(pp) as f:
            for ln in f:
                parts = ln.split()
                if not parts: continue
                if parts[0] == 'beta' and len(parts) >= 3:
                    h_beta = float(parts[2])
                elif parts[0] == 'mu' and len(parts) >= 3:
                    c = int(parts[1])
                    if c < D: h_mu[c] = float(parts[2])
                elif parts[0] == 'alpha' and len(parts) >= 4:
                    c = int(parts[1]); j = int(parts[2])
                    if c < D and j < D:
                        h_alpha[c, j] = float(parts[3])

    # phi has shape (D,) — one memory state per event type.
    phi_per_seed = np.zeros((len(seeds), D), dtype=np.float64)
    phi = np.zeros(D, dtype=np.float64)
    ev_i = 0
    prev_t = 0
    for k_seed, srow in enumerate(seed_row):
        while ev_i < len(ev_t) and ev_t[ev_i] <= srow:
            t_ev = ev_t[ev_i]
            dt = t_ev - prev_t
            if dt > 0: phi *= np.exp(-h_beta * dt)
            c = ev_pt[ev_i]
            if 0 <= c < D: phi[c] += 1.0
            prev_t = t_ev
            ev_i += 1
        dt = srow - prev_t
        phi_per_seed[k_seed] = phi * np.exp(-h_beta * dt) if dt > 0 else phi.copy()

    phi_feats = [phi_per_seed[:, i:i+1] for i in range(D)]

    # α·φ rate: λ_c = μ_c + Σ_j α_{c,j} φ_j
    rates = phi_per_seed @ h_alpha.T + h_mu[None, :]
    rate_feats = [rates[:, i:i+1] for i in range(D)]

    # Micro-products (sim-state derivable — functions of current book)
    micro_feats = [
        (imb0 * sp)[:, None],                    # antisym × sym = antisym
        (imb1 * sp)[:, None],                    # antisym
        (imb0 * imb1)[:, None],                  # antisym × antisym = sym
        (imb0_S * sp)[:, None],                  # antisym
    ]
    micro_signs = [-1, -1, +1, -1]               # 3 antisym, 1 sym

    X = np.hstack(base_sym + phi_feats + rate_feats
                  + base_anti + micro_feats).astype(np.float32)

    # Layout: base_sym(13) | phi(D=6, sym) | rate(D=6, sym) | base_anti(12) | micro(4)
    # Hawkes phi/rate are pooled (no ask/bid asymmetry), treated as sym under mirror.
    signs = np.ones(X.shape[1], dtype=np.float32)
    o = n_sym_base + 2 * D                       # skip base_sym + phi + rate (all sym)
    signs[o : o + n_anti_base] = -1              # base_anti
    o += n_anti_base
    for i, s_m in enumerate(micro_signs):
        signs[o + i] = s_m

    ys = {}
    for T in T_LIST:
        fut = seeds + T
        fut_mid = (books[fut, 0] + books[fut, 8]) / 2.0
        ys[T] = (fut_mid - mid).astype(np.float32)
    return X, ys, signs

offs = np.fromfile('data/sessions.events.raw', dtype=np.int64)
total_bytes = os.path.getsize('data/train.events')
ev_mm = np.memmap('data/train.events', dtype=np.int32, mode='r',
                  shape=(total_bytes // RECSZ, 54))

t0 = time.time()
n_feats = None
last_signs = None
for s in range(62):
    r = load_session(s, offs, ev_mm)
    if r is None:
        print(f"  ses{s}: empty — skipping"); continue
    X, ys, signs = r
    np.savez(f'{OUTDIR}/s{s}.npz', X=X, y1=ys[1], y5=ys[5], y20=ys[20], y55=ys[55])
    if n_feats is None:
        n_feats = X.shape[1]
        print(f"  N_feats = {n_feats}")
    last_signs = signs
np.savez(f'{OUTDIR}/mirror.npz', perm=np.arange(n_feats), signs=last_signs)
print(f"  n_sym  = {int((last_signs == +1).sum())}")
print(f"  n_anti = {int((last_signs == -1).sum())}")
print(f"built features for 62 sessions in {time.time()-t0:.1f}s")
