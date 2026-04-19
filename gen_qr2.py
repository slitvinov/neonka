"""Generate QR2 tables with opposite-queue bucketing (Huang-Lehalle-Rosenbaum).

Rate(sp, n_own, opp_bucket) per (event, side).  Opp bucket: {0: 0, 1: 1-2, 2: 3-5, 3: 6+}.
Writes /tmp/neonka/tables/{S}/qr2.{tp,tm}.{a,b}.rates with lines 'sp n_own opp rate'.
"""
import numpy as np
import os, sys

SP_MAX, N0_MAX, OPP = 64, 16, 4


def opp_bucket(n):
    if n == 0: return 0
    if n <= 2: return 1
    if n <= 5: return 2
    return 3


def process(sid):
    offs = np.fromfile('data/sessions.raw', dtype=np.int64)
    r = np.memmap('data/train.raw', dtype=np.int32, mode='r').reshape(-1, 49)
    r = r[int(offs[sid]):int(offs[sid+1])]
    if len(r) < 100: return None
    sp = (r[:-1, 0] - r[:-1, 8]).astype(np.int64)
    aN0 = r[:-1, 32]; bN0 = r[:-1, 40]
    aR0a = r[:-1, 0]; aR0b = r[1:, 0]
    bR0a = r[:-1, 8]; bR0b = r[1:, 8]
    aN0b = r[1:, 32]; bN0b = r[1:, 40]

    tp_a = (aR0b < aR0a) | ((aR0b == aR0a) & (aN0b > aN0))
    tm_a = (aR0b > aR0a) | ((aR0b == aR0a) & (aN0b < aN0))
    tp_b = (bR0b > bR0a) | ((bR0b == bR0a) & (bN0b > bN0))
    tm_b = (bR0b < bR0a) | ((bR0b == bR0a) & (bN0b < bN0))

    sp_c = np.clip(sp, 0, SP_MAX-1).astype(int)
    an_c = np.clip(aN0, 0, N0_MAX-1).astype(int)
    bn_c = np.clip(bN0, 0, N0_MAX-1).astype(int)
    opp_a_vals = np.array([opp_bucket(int(b)) for b in bN0], dtype=int)
    opp_b_vals = np.array([opp_bucket(int(a)) for a in aN0], dtype=int)

    cnt_tp_a = np.zeros((SP_MAX, N0_MAX, OPP), np.int64)
    cnt_tm_a = np.zeros((SP_MAX, N0_MAX, OPP), np.int64)
    cnt_tp_b = np.zeros((SP_MAX, N0_MAX, OPP), np.int64)
    cnt_tm_b = np.zeros((SP_MAX, N0_MAX, OPP), np.int64)
    ntics_a  = np.zeros((SP_MAX, N0_MAX, OPP), np.int64)
    ntics_b  = np.zeros((SP_MAX, N0_MAX, OPP), np.int64)

    np.add.at(ntics_a, (sp_c, an_c, opp_a_vals), 1)
    np.add.at(ntics_b, (sp_c, bn_c, opp_b_vals), 1)
    np.add.at(cnt_tp_a, (sp_c, an_c, opp_a_vals), tp_a)
    np.add.at(cnt_tm_a, (sp_c, an_c, opp_a_vals), tm_a)
    np.add.at(cnt_tp_b, (sp_c, bn_c, opp_b_vals), tp_b)
    np.add.at(cnt_tm_b, (sp_c, bn_c, opp_b_vals), tm_b)

    D = f'/tmp/neonka/tables/{sid}'
    os.makedirs(D, exist_ok=True)
    for key, cnt, ntics in [('tp_a', cnt_tp_a, ntics_a),
                            ('tm_a', cnt_tm_a, ntics_a),
                            ('tp_b', cnt_tp_b, ntics_b),
                            ('tm_b', cnt_tm_b, ntics_b)]:
        ev, side = key.split('_')
        with open(f'{D}/qr2.{ev}.{side}.rates', 'w') as f:
            for s in range(SP_MAX):
                for a in range(N0_MAX):
                    for o in range(OPP):
                        if ntics[s, a, o] < 5: continue
                        f.write(f'{s} {a} {o} {cnt[s, a, o] / ntics[s, a, o]:g}\n')
    return ntics_a


if __name__ == '__main__':
    for sid in range(62):
        nt = process(sid)
        if sid == 45 and nt is not None:
            print(f'ses45 ntics_a count at sp=8 × (n_own × opp_bucket):')
            for a in range(1, 6):
                row = ' '.join(f'{nt[8, a, o]:>6}' for o in range(OPP))
                print(f'  n_own={a}: {row}')
    print('\nDone.')
