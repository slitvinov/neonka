import sys, time
import numpy as np
import torch
import torch.nn as nn

SES = int(sys.argv[1]) if len(sys.argv) > 1 else 0
K   = int(sys.argv[2]) if len(sys.argv) > 2 else 64
STR = int(sys.argv[3]) if len(sys.argv) > 3 else 10
EPO = int(sys.argv[4]) if len(sys.argv) > 4 else 30
MOD = sys.argv[5] if len(sys.argv) > 5 else "tx"

r = np.fromfile('data/train.raw', dtype=np.int32).reshape(-1, 49)
b = np.fromfile('data/sessions.raw', dtype=np.int64)
lo, hi = int(b[SES]), int(b[SES+1])
x = r[lo:hi]

aR, bR = x[:, 0].astype(float), x[:, 8].astype(float)
aN0, bN0 = x[:, 32].astype(float), x[:, 40].astype(float)
aN1, bN1 = x[:, 33].astype(float), x[:, 41].astype(float)
aN2, bN2 = x[:, 34].astype(float), x[:, 42].astype(float)
sp   = aR - bR
mid  = (aR + bR) / 2.0
imb0 = (aN0 - bN0) / np.maximum(aN0 + bN0, 1)
imb1 = (aN1 - bN1) / np.maximum(aN1 + bN1, 1)
imb2 = (aN2 - bN2) / np.maximum(aN2 + bN2, 1)
dmid1 = np.concatenate([[0.], mid[1:] - mid[:-1]])
dsp1  = np.concatenate([[0.], sp[1:]  - sp[:-1]])

feats = np.stack([sp, imb0, imb1, imb2,
                  np.log1p(aN0), np.log1p(bN0), np.log1p(aN1), np.log1p(bN1),
                  dmid1, dsp1], 1).astype(np.float32)
y = x[:, 48].astype(np.float32) / 4.0
D = feats.shape[1]

valid = np.arange(K, len(x), STR)
Xs = np.stack([feats[t-K:t] for t in valid])
Ys = y[valid]

half = len(Xs) // 2
Xtr, Ytr = Xs[:half], Ys[:half]
Xev, Yev = Xs[half:], Ys[half:]

mean = Xtr.mean(axis=(0,1))
std  = Xtr.std(axis=(0,1)) + 1e-6
Xtr = (Xtr - mean) / std
Xev = (Xev - mean) / std
ym, ys = Ytr.mean(), Ytr.std() + 1e-6

class TxModel(nn.Module):
    def __init__(self, d_in, d_model=16, nhead=2, nlayers=1, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.pos  = nn.Parameter(torch.zeros(1, K, d_model))
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=32,
                                         batch_first=True, dropout=dropout)
        self.tx = nn.TransformerEncoder(enc, nlayers)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)
    def forward(self, x):
        h = self.proj(x) + self.pos
        h = self.tx(h)
        h = h.mean(dim=1)
        return self.head(self.drop(h)).squeeze(-1)

class MLP(nn.Module):
    def __init__(self, d_in, hidden=32, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in * K, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))
    def forward(self, x):
        return self.net(x.reshape(x.shape[0], -1)).squeeze(-1)

class LastMLP(nn.Module):
    def __init__(self, d_in, hidden=32, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))
    def forward(self, x):
        return self.net(x[:, -1]).squeeze(-1)

dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
model = {"tx": TxModel, "mlp": MLP, "last": LastMLP}[MOD](D).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPO)

n_par = sum(p.numel() for p in model.parameters())
print(f"model={MOD} params={n_par}  samples train={len(Xtr)} eval={len(Xev)}")

Xtr_t = torch.from_numpy(Xtr).to(dev)
Ytr_t = torch.from_numpy((Ytr - ym) / ys).to(dev)
Xev_t = torch.from_numpy(Xev).to(dev)

def fit_eval(Ft, Yt, Fe, Ye, lam=1.0):
    A = np.hstack([Ft, np.ones((len(Ft),1))])
    w = np.linalg.solve(A.T @ A + lam*np.eye(A.shape[1]), A.T @ Yt)
    yh = np.hstack([Fe, np.ones((len(Fe),1))]) @ w
    return 1 - ((Ye-yh)**2).sum() / ((Ye-Ye.mean())**2).sum()

r2_last = fit_eval(Xtr[:,-1], Ytr, Xev[:,-1], Yev)
r2_flat = fit_eval(Xtr.reshape(len(Xtr),-1), Ytr, Xev.reshape(len(Xev),-1), Yev, lam=100.)
print(f"[baseline] Ridge last-frame  : test R²={100*r2_last:+.3f}%")
print(f"[baseline] Ridge seq-flat    : test R²={100*r2_flat:+.3f}%")

t0 = time.time()
BS = 256
best_r2 = -1e9
for epoch in range(EPO):
    model.train()
    perm = torch.randperm(len(Xtr_t))
    tot = 0
    for i in range(0, len(perm), BS):
        idx = perm[i:i+BS]
        xb, yb = Xtr_t[idx], Ytr_t[idx]
        pred = model(xb)
        loss = nn.functional.mse_loss(pred, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(idx)
    sched.step()
    model.eval()
    with torch.no_grad():
        pe = model(Xev_t).cpu().numpy() * ys + ym
    r2 = 1 - ((Yev - pe)**2).sum() / ((Yev - Yev.mean())**2).sum()
    corr = np.corrcoef(Yev, pe)[0,1]
    if r2 > best_r2: best_r2 = r2
    if (epoch % 3) == 0 or epoch == EPO - 1:
        print(f"epoch {epoch:2d}  train={tot/len(Xtr):.4f}  test R²={100*r2:+.3f}%  corr={corr:+.3f}  best={100*best_r2:+.3f}%")

print(f"elapsed: {time.time()-t0:.1f}s  device={dev}  model={MOD}  ses={SES}  K={K}  stride={STR}")
