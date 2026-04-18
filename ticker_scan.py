"""Scan many LSE stocks for time-windowed match to our 63 session medians.
Allow time offset. Skip shift (unrealistic). Report top matches by RMSE+corr.
"""
import sys, subprocess, numpy as np

try: import yfinance as yf
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "--break-system-packages", "--user", "yfinance", "-q"])
    import yfinance as yf

our = np.array([float(open(f"/tmp/lsegscan/_med_{s}.txt").read()) for s in range(63)])
print(f"ours: n=63 med={np.median(our):.0f} range=[{our.min():.0f},{our.max():.0f}]")

TICKERS = ["LSEG.L","NXT.L","AZN.L","LMP.L","BNZL.L","CPG.L","DCC.L","WTB.L","PSN.L","RKT.L",
           "ULVR.L","DGE.L","HLMA.L","REL.L","BARC.L","AAL.L","III.L","ITRK.L","SN.L","CRDA.L",
           "SPX.L","AV.L","SGE.L","ADM.L","PSON.L","INF.L","STAN.L","BA.L","LGEN.L","ABF.L",
           "JD.L","SMIN.L","SSE.L","IMB.L","PRU.L","MNDI.L","BT-A.L","GLEN.L","ANTO.L"]

results = []
for tk in TICKERS:
    try:
        df = yf.download(tk, start="2019-01-01", end="2026-04-17", progress=False, auto_adjust=False)
        if len(df) < 100: continue
        close = df['Close'].to_numpy().flatten()
        N = 63
        best_rmse = 1e18
        best_info = None
        for off in range(0, len(close)-N):
            w = close[off:off+N]
            if np.std(w) < 10: continue
            rmse = float(np.sqrt(np.mean((our-w)**2)))
            corr = float(np.corrcoef(our,w)[0,1]) if np.std(w)>0 else -1
            if rmse < best_rmse:
                best_rmse = rmse
                best_info = (corr, df.index[off].date(), df.index[off+N-1].date(), w.min(), w.max())
        if best_info:
            results.append((best_rmse, tk, *best_info))
    except Exception as e:
        pass

results.sort()
print(f"\n{'ticker':<10} {'RMSE':>7} {'corr':>6} {'range':>20} {'dates':>30}")
for r in results[:15]:
    rmse, tk, corr, d0, d1, mn, mx = r
    print(f"{tk:<10} {rmse:>7.0f} {corr:>+6.3f} [{mn:>5.0f},{mx:>5.0f}]     {d0} → {d1}")
