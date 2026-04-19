"""Fold the per-side imb-bin rate tables into mirror-symmetric pooled tables.

Under mirror (ask↔bid swap): imb_bin 0↔5, 1↔4, 2↔3.  Enforce:
  new_a[i] = 0.5 * (a[i]     + b[mirror(i)])
  new_b[i] = 0.5 * (a[mirror(i)] + b[i])

After pooling, a[i] == b[mirror(i)] and total rate at imb=i equals total at
mirror(i).  Physical per-side asymmetry inside each bin pair is preserved.

Usage: python3 pool_tables.py /tmp/tables<S>
"""
import os, sys

MIRROR = {0: 5, 1: 4, 2: 3, 3: 2, 4: 1, 5: 0}
EVENTS = ['tp', 'tm', 'tm_q', 'tm_c', 'dp', 'dm']


def load_kv(path):
    if not os.path.exists(path): return {}
    d = {}
    for l in open(path):
        parts = l.split()
        if len(parts) == 2:
            d[int(float(parts[0]))] = float(parts[1])
    return d


def save_kv(path, d):
    with open(path, 'w') as f:
        for k in sorted(d):
            f.write(f'{k} {d[k]:g}\n')


def pool(directory, ev):
    orig_a = [load_kv(f'{directory}/{ev}.a.imb{i}.rates') for i in range(6)]
    orig_b = [load_kv(f'{directory}/{ev}.b.imb{i}.rates') for i in range(6)]
    new_a = [None] * 6
    new_b = [None] * 6
    for i in range(6):
        m = MIRROR[i]
        keys_a = set(orig_a[i]) | set(orig_b[m])
        keys_b = set(orig_a[m]) | set(orig_b[i])
        new_a[i] = {k: 0.5 * (orig_a[i].get(k, 0) + orig_b[m].get(k, 0)) for k in keys_a}
        new_b[i] = {k: 0.5 * (orig_a[m].get(k, 0) + orig_b[i].get(k, 0)) for k in keys_b}
    for i in range(6):
        save_kv(f'{directory}/{ev}.a.imb{i}.rates', new_a[i])
        save_kv(f'{directory}/{ev}.b.imb{i}.rates', new_b[i])


def pool_n(directory):
    # n.imbX.rates is (other-event fraction); fold by averaging mirror pairs.
    tabs = [load_kv(f'{directory}/n.imb{i}.rates') for i in range(6)]
    new = [None] * 6
    for i in range(6):
        m = MIRROR[i]
        keys = set(tabs[i]) | set(tabs[m])
        new[i] = {k: 0.5 * (tabs[i].get(k, 0) + tabs[m].get(k, 0)) for k in keys}
    for i in range(6):
        save_kv(f'{directory}/n.imb{i}.rates', new[i])


if __name__ == '__main__':
    d = sys.argv[1]
    for ev in EVENTS: pool(d, ev)
    pool_n(d)
    print(f'pooled {d}')
