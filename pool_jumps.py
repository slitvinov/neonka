"""Merge per-session histogram tables (tp/dp jumps, refill) into a shared
common dir by SUMMING counts bin-by-bin.

Design (parallel-safe by construction):
  • Each worker writes its own per-session file (/tmp/tables{S}/*) — no
    contention, no locking required.
  • This script is a *separate, serial* merge step run afterwards.  Each
    pooled bin is the sum of the same-bin count across all 62 sessions.
  • No symlinks; per-session files are left in place (workers can re-run
    any session independently without touching the pool).

Pools:
  tp.own, dp.own                 — global jump dists (shape CV ≈ 0.12, 0.41)
  tp.own.spN, dp.own.spN         — sp-conditional jump dists (CV ≈ 0.16-0.34)
  refill.a.own, refill.b.own     — refill dists (CV ≈ 0.2 in main mode)

Output: /tmp/neonka/tables/common/ — pass to onestep via `-g /tmp/neonka/tables/common`.

Keeps the rate tables (tp/tm_q/tm_c/dp/dm × a,b × imb) per-session — those
have regime-dependent CV ~0.7-0.9 and should not be pooled.
"""
import os, glob, sys
from collections import defaultdict

COMMON_DIR = '/tmp/neonka/tables/common'
os.makedirs(COMMON_DIR, exist_ok=True)
POOL_NAMES = ['tp.own', 'dp.own', 'refill.a.own', 'refill.b.own']
POOL_PATTERNS = ['tp.own.sp*', 'dp.own.sp*']


def load_counts(p):
    if not os.path.isfile(p): return {}
    d = {}
    for l in open(p):
        parts = l.split()
        if len(parts) == 2:
            d[int(float(parts[0]))] = int(float(parts[1]))
    return d


def save_counts(p, d):
    with open(p, 'w') as f:
        for k in sorted(d):
            f.write(f'{k} {d[k]}\n')


for name in POOL_NAMES:
    merged = defaultdict(int)
    n_sessions = 0
    for s in range(62):
        kv = load_counts(f'/tmp/neonka/tables/{s}/{name}')
        if kv:
            n_sessions += 1
            for k, v in kv.items(): merged[k] += v
    if merged:
        save_counts(f'{COMMON_DIR}/{name}', dict(merged))
    print(f'{name}: {n_sessions} sessions, {sum(merged.values())} total events, '
          f'{len(merged)} bins')

for pat in POOL_PATTERNS:
    by_sp = defaultdict(lambda: defaultdict(int))
    for s in range(62):
        for f in glob.glob(f'/tmp/neonka/tables/{s}/{pat}'):
            sp = int(f.rsplit('.sp', 1)[1])
            for k, v in load_counts(f).items():
                by_sp[sp][k] += v
    for sp, h in by_sp.items():
        name = pat.replace('*', str(sp))
        save_counts(f'{COMMON_DIR}/{name}', dict(h))
    print(f'{pat}: {len(by_sp)} sp values, '
          f'{sum(sum(h.values()) for h in by_sp.values())} total events')

print(f'\nmerged into {COMMON_DIR}')
os.system(f'du -sh {COMMON_DIR}')
