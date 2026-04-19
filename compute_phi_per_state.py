"""Compute E[φ_c | state=(sp, imb)] per session.  Fixes the bug where
rate tables were averaged without Hawkes awareness.

Method:
  1. Walk train.events for each session; track φ_c via exponential decay.
  2. At each IDLE marker (row), record current (sp, imb_bin) and φ snapshot.
  3. Aggregate: per state, mean φ_c across all visits.

Output: /tmp/neonka/hawkes/{S}.phi_s  — lines 'sp imb phi_0 phi_1 ... phi_5'
"""
import os, sys
import numpy as np

D = 6
HK_BETA = 0.05  # matches onestep single-β

def pooled_type(t, aN0, bN0):
    if t == 0 or t == 1: return 0
    if t == 2: return 1 if aN0 > 1 else 2
    if t == 3: return 1 if bN0 > 1 else 2
    if t == 4 or t == 5: return 3
    if t == 6 or t == 7: return 4
    if t >= 9: return 5
    return -1

def imb_bin(aN0, bN0, aN1, bN1):
    s = int(aN0) + int(bN0); d = int(aN0) - int(bN0)
    b0 = 1 if s == 0 else (0 if d*5 < -s else (2 if d*5 > s else 1))
    return b0 * 2 + (1 if aN1 > bN1 else 0)

# Load events
offs_bytes = np.fromfile('data/sessions.events.raw', dtype=np.int64)
REC_SIZE = 54 * 4
ev_mm = np.memmap('data/train.events', dtype=np.int32, mode='r').reshape(-1, 54)
n_sess = len(offs_bytes) - 1

for S in range(n_sess):
    lo = int(offs_bytes[S]) // REC_SIZE
    hi = int(offs_bytes[S+1]) // REC_SIZE
    block = ev_mm[lo:hi]
    if len(block) < 100: continue

    # Tracking: φ per type, last update time
    phi = np.zeros(D)
    last_t = int(block[0, 1])
    # Per-state accumulator: sum(φ), count
    sum_phi = {}  # (sp, imb) -> np.array(D)
    count = {}

    for row in block:
        t_type = int(row[0])
        t_row  = int(row[1])
        # Decay φ to t_row
        dt = t_row - last_t
        if dt > 0:
            phi *= np.exp(-HK_BETA * dt)
            last_t = t_row
        if t_type == 8:        # IDLE = row snapshot
            aN0 = int(row[5 + 32]); bN0 = int(row[5 + 40])
            aN1 = int(row[5 + 33]); bN1 = int(row[5 + 41])
            aR0 = int(row[5 + 0]);  bR0 = int(row[5 + 8])
            sp = aR0 - bR0
            if 0 <= sp < 64:
                im = imb_bin(aN0, bN0, aN1, bN1)
                key = (sp, im)
                if key not in sum_phi:
                    sum_phi[key] = np.zeros(D)
                    count[key] = 0
                sum_phi[key] += phi
                count[key] += 1
        else:
            # Accumulate event at its pre-event state
            aN0 = int(row[5 + 32]); bN0 = int(row[5 + 40])
            c = pooled_type(t_type, aN0, bN0)
            if 0 <= c < D:
                phi[c] += 1.0

    # Write per-state φ means
    out = f'/tmp/neonka/hawkes/{S}.phi_s'
    with open(out, 'w') as f:
        for (sp, im), total_phi in sum_phi.items():
            n = count[(sp, im)]
            if n < 5: continue
            avg = total_phi / n
            f.write(f'{sp} {im} ' + ' '.join(f'{v:g}' for v in avg) + '\n')
    if S == 45:
        print(f'ses{S}: {len(sum_phi)} state bins')
        # Print a few entries at sp=8 to verify
        for im in range(6):
            k = (8, im)
            if k in sum_phi:
                avg = sum_phi[k] / count[k]
                print(f'  sp=8 imb={im}: n={count[k]:>6}  E[φ]=[{", ".join(f"{v:.3f}" for v in avg)}]')

print(f'\nWrote phi_s tables for {n_sess} sessions')
