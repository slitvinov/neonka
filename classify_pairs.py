"""Classify pairs by elementary event count. Reveals if data is tick-quantized or
clustered. Compares real session vs a sim chain trajectory.

Usage:
  python3 classify_pairs.py [session_id] [n_frames_sim]
"""
import sys, subprocess, numpy as np

S         = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N_SIM     = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
N_CLASSIFY = 10000

r = np.fromfile('data/train.raw', dtype=np.int32).reshape(-1, 49)
b = np.fromfile('data/sessions.raw', dtype=np.int64)
lo, hi = int(b[S]), int(b[S+1])
real = r[lo:hi]

seed = real[:1].tobytes()
tmp_seed = '/tmp/classify_seed.raw'
tmp_chain = '/tmp/classify_chain.raw'
with open(tmp_seed, 'wb') as f:
    f.write(seed)
D = f"data/tables/{S:02d}"
subprocess.run(['./onestep', '-m', D, '-T', '1', '-N', str(N_SIM), '-R', '1', '-P'],
               stdin=open(tmp_seed, 'rb'),
               stdout=open(tmp_chain, 'wb'))
sim = np.fromfile(tmp_chain, dtype=np.int32).reshape(-1, 49)

def walk(R0, N0, R1, N1, diff):
    """Merge-walk that counts total unit events between pair (prev, cur).
    diff: +1 for ask (prices ascend with level), -1 for bid (descend).
    Returns sum of |ΔN| across all merged levels — i.e., count of elementary events.
    """
    i = j = 0
    n = 0
    while i < 8 and j < 8 and N0[i] != 0 and N1[j] != 0:
        d = diff * (R1[j] - R0[i])
        if d < 0:
            n += int(N1[j]); j += 1
        elif d == 0:
            dn = int(N1[j]) - int(N0[i])
            if dn: n += abs(dn)
            i += 1; j += 1
        else:
            n += int(N0[i]); i += 1
    return n

def classify(arr, n):
    ev = np.empty(n, dtype=np.int32)
    for k in range(n):
        prev, cur = arr[k], arr[k+1]
        a = walk(prev[0:8],  prev[32:40], cur[0:8],  cur[32:40], +1)
        b = walk(prev[8:16], prev[40:48], cur[8:16], cur[40:48], -1)
        ev[k] = a + b
    return ev

def report(ev, name):
    print(f"\n{name}  (N pairs analyzed: {len(ev)})")
    u, c = np.unique(ev, return_counts=True)
    for uu, cc in zip(u[:8], c[:8]):
        print(f"  {uu} events: {100*cc/len(ev):5.1f}%  ({cc})")
    if (ev >= 3).any():
        print(f"  3+ events: {100*(ev >= 3).mean():5.1f}%")
    print(f"  mean events/pair: {ev.mean():.3f}")

n_real = min(N_CLASSIFY, len(real)-1)
n_sim  = min(N_CLASSIFY, len(sim)-1)
report(classify(real, n_real), f"Real session {S}")
report(classify(sim,  n_sim),  "Sim chain (Poisson)")
