import numpy as np

D, BETA, RHO_MAX, SESSION = 5, 0.05, 0.9, 0

REC = 54 * 4
offs = np.fromfile('data/sessions.events.raw', dtype=np.int64)
ev = np.memmap('data/train.events', dtype=np.int32, mode='r').reshape(-1, 54)
lo = int(offs[SESSION]) // REC
hi = int(offs[SESSION + 1]) // REC
block = ev[lo:hi]

def pooled(t, aN0, bN0):
    if t in (0, 1): return 0
    if t == 2: return 1 if aN0 > 1 else 2
    if t == 3: return 1 if bN0 > 1 else 2
    if t in (4, 5): return 3
    if t in (6, 7): return 4
    return -1

times = []; types = []
for row in block:
    t_type = int(row[0])
    if t_type == 8: continue
    c = pooled(t_type, int(row[5 + 32]), int(row[5 + 40]))
    if 0 <= c < D:
        times.append(int(row[1])); types.append(c)
times = np.array(times, dtype=np.int64)
types = np.array(types, dtype=np.int32)
N, T_total = len(times), times[-1] - times[0] + 1
labels = ['tp', 'tm_q', 'tm_c', 'dp', 'dm']
counts = np.bincount(types, minlength=D)
print(f'N={N} T={T_total} rate={N/T_total:.4f}')
print(f'marginals:', dict(zip(labels, (counts/N).round(4))))

phi_at = np.zeros((N, D))
phi = np.zeros(D)
last_t = times[0]
for i in range(N):
    dt = times[i] - last_t
    if dt > 0: phi *= np.exp(-BETA * dt)
    phi_at[i] = phi
    phi[types[i]] += 1
    last_t = times[i]

G = np.zeros(D)
for i in range(N):
    G[types[i]] += (1 - np.exp(-BETA * (T_total - (times[i] - times[0])))) / BETA

mu = counts / T_total * 0.5
alpha = np.ones((D, D)) * 0.005 * BETA
for it in range(200):
    ap = np.einsum('ij,ij->i', alpha[types], phi_at)
    lam = np.maximum(mu[types] + ap, 1e-12)
    w_mu = mu[types] / lam
    w_a = (alpha[types] * phi_at) / lam[:, None]
    mu_new = np.zeros(D); a_new = np.zeros((D, D))
    for c in range(D):
        m = types == c
        if m.sum() > 0:
            mu_new[c] = w_mu[m].sum() / T_total
            a_new[c] = w_a[m].sum(axis=0) / np.maximum(G, 1e-12)
    for c in range(D):
        r = a_new[c].sum() / BETA
        if r > RHO_MAX: a_new[c] *= RHO_MAX / r
    mu, alpha = mu_new, a_new

print(f'mu={mu.round(4)}')
print(f'rho_c={(alpha/BETA).sum(axis=1).round(3)}')

real_tc = np.bincount((times - times[0]).astype(int), minlength=T_total)
pmf = np.zeros(6)
for k in range(6): pmf[k] = (real_tc == k).mean()
pmf[5] = (real_tc >= 5).mean()

np.random.seed(42)
sim_times = []; sim_types = []
phi = np.zeros(D)
dec = np.exp(-BETA)
cum_pmf = np.cumsum(pmf)
t0 = int(times[0])
for tk in range(T_total):
    K = np.searchsorted(cum_pmf, np.random.rand())
    for _ in range(K):
        lam = np.maximum(mu + alpha @ phi, 1e-8)
        c = np.random.choice(D, p=lam / lam.sum())
        sim_times.append(t0 + tk); sim_types.append(c)
        phi[c] += 1
    phi *= dec
sim_times = np.array(sim_times); sim_types = np.array(sim_types)

def trans_matrix(ts, lag=1):
    M = np.zeros((D, D))
    if len(ts) <= lag: return M
    for cp in range(D):
        m = ts[:-lag] == cp
        if m.sum() > 0:
            M[cp] = np.bincount(ts[lag:][m], minlength=D) / m.sum()
    return M

sim_tc = np.bincount(sim_times - sim_times[0], minlength=T_total)
print(f'\nevents per tick:')
print(f'  k | real%   sim%')
for k in range(6):
    print(f'  {k} | {(real_tc==k).mean()*100:>6.2f}  {(sim_tc==k).mean()*100:>6.2f}')

mg_real = counts / N
mg_sim = np.bincount(sim_types, minlength=D) / max(len(sim_types), 1)
print(f'\nmarginals:')
for c in range(D):
    print(f'  {labels[c]:>5}: real={mg_real[c]:.4f} sim={mg_sim[c]:.4f}')

T1_r = trans_matrix(types, 1)
T1_s = trans_matrix(sim_types, 1)
print(f'\nlag-1 P(c_t|c_t-1):')
print(f'  prev |', ' '.join(f'{l:>6}' for l in labels))
for c in range(D):
    print(f'  {labels[c]:>4} R|', ' '.join(f'{T1_r[c,b]:>6.3f}' for b in range(D)))
    print(f'       S|', ' '.join(f'{T1_s[c,b]:>6.3f}' for b in range(D)),
          f'  max|Δ|={np.abs(T1_r[c]-T1_s[c]).max():.3f}')

gaps_r = np.diff(times.astype(float))
gaps_s = np.diff(sim_times.astype(float))
print(f'\ngaps: real mean={gaps_r.mean():.3f} std={gaps_r.std():.3f} CV={gaps_r.std()/gaps_r.mean():.3f}')
print(f'      sim  mean={gaps_s.mean():.3f} std={gaps_s.std():.3f} CV={gaps_s.std()/gaps_s.mean():.3f}')
