"""Wide ticker search: LSE, NYSE, NASDAQ, ETFs, futures.
Allow free time offset, optional price shift (report both, shift <=5%).
"""
import sys, subprocess, numpy as np
try: import yfinance as yf
except ImportError:
    subprocess.run([sys.executable,"-m","pip","install","--break-system-packages","--user","yfinance","-q"])
    import yfinance as yf

our = np.array([float(open(f"/tmp/lsegscan/_med_{s}.txt").read()) for s in range(63)])
print(f"ours: n=63 med={np.median(our):.0f} range=[{our.min():.0f},{our.max():.0f}]")

TICKERS = (
  "LSEG.L NXT.L AZN.L LMP.L BNZL.L CPG.L DCC.L WTB.L PSN.L RKT.L ULVR.L DGE.L HLMA.L "
  "REL.L BARC.L AAL.L III.L ITRK.L SN.L CRDA.L SPX.L AV.L SGE.L ADM.L PSON.L INF.L "
  "STAN.L BA.L LGEN.L ABF.L JD.L SMIN.L SSE.L IMB.L PRU.L MNDI.L BT-A.L GLEN.L ANTO.L "
  "EXPN.L FLTR.L HSBA.L LAND.L RIO.L RR.L SGRO.L SVT.L TSCO.L UU.L VOD.L MRO.L SKG.L "
  "BARC.L LLOY.L NWG.L CNA.L BRBY.L TW.L WEIR.L FRES.L EVR.L ITV.L ICP.L SDR.L PHNX.L "
  "AAPL MSFT GOOGL AMZN META NVDA TSLA BRK-B JPM V MA DIS NFLX ADBE CRM ORCL PEP KO "
  "WMT HD UNH LLY XOM CVX PFE ABBV MRK JNJ PG MCD NKE SBUX CAT DE BA GE F GM "
  "UPS FDX CSCO INTC AMD QCOM TXN AVGO NOW AMAT MU KLAC LRCX "
  "SPY QQQ DIA IWM EFA EEM VTI VOO IVV GLD SLV USO TLT HYG LQD "
  "EWU EWJ EWG EWQ EWC EZU FXI INDA EWY EWA EWH VEA VWO "
  "ABNB ADI ADP AIG AMGN AMT AON AXP BAC BK BLK BMY BX C CB CI CL COF COP COST CRM "
  "CSX CVS DD DHR DIS DOW DUK EL EMR EOG EPD ETN EW F FDX FIS FISV GD GE GILD GS HAL "
  "HCA HON IBM ICE ILMN INTU ISRG ITW KHC KMB KMI KO LIN LMT LOW LYB MA MCD MCO MDLZ "
  "MDT MET MMC MMM MO MPC MRNA MS MSI MU NEE NEM NKE NOC NOW NSC NUE OXY PANW PEP "
  "PFE PG PGR PH PLD PM PNC PPG PRU PSA PSX PYPL QCOM RTX SBUX SCHW SHW SLB SO SPGI "
  "SYK T TFC TGT TJX TMO TMUS TRV TSN TT TXN UNP UPS USB V VLO VZ WBA WFC WM WMT XOM"
).split()

seen = set(); uniq = []
for t in TICKERS:
    if t not in seen: seen.add(t); uniq.append(t)

import os, pickle
os.makedirs("/tmp/yfcache", exist_ok=True)
def scan_one(tk):
    try:
        cache = f"/tmp/yfcache/{tk.replace('/','_')}.pkl"
        if os.path.exists(cache):
            import pickle
            with open(cache,"rb") as f: df = pickle.load(f)
        else:
            df = yf.download(tk, start="2015-01-01", end="2026-04-17",
                             progress=False, auto_adjust=False, timeout=15)
            with open(cache,"wb") as f: pickle.dump(df, f)
        if df is None or len(df) < 63: return None
        close = df['Close'].to_numpy().flatten()
        dates = df.index
        N = 63
        best = (1e18, None)
        best_sh = (1e18, None)
        for off in range(0, len(close)-N):
            w = close[off:off+N]
            if np.std(w) < 1e-6: continue
            rmse = float(np.sqrt(np.mean((our-w)**2)))
            corr = float(np.corrcoef(our,w)[0,1])
            if rmse < best[0]:
                best = (rmse, (off, corr, dates[off].date(), dates[off+N-1].date(), w.min(), w.max()))
            shift = float(np.median(our) - np.median(w))
            if abs(shift) / np.median(w) < 0.05:
                rmse_sh = float(np.sqrt(np.mean((our-(w+shift))**2)))
                if rmse_sh < best_sh[0]:
                    best_sh = (rmse_sh, (off, corr, shift, dates[off].date(), dates[off+N-1].date()))
        return (tk, best, best_sh)
    except Exception as e:
        print(f"  ERR {tk}: {e}", file=sys.stderr)
        return None

noshift, shift5 = [], []
for i, tk in enumerate(uniq):
    res = scan_one(tk)
    if res is None: continue
    tk2, b, bs = res
    if b[1]: noshift.append((b[0], tk2, *b[1]))
    if bs[1]: shift5.append((bs[0], tk2, *bs[1]))
    if i % 20 == 0: print(f"  {i}/{len(uniq)} noshift={len(noshift)} shift={len(shift5)}", file=sys.stderr, flush=True)

noshift.sort(); shift5.sort()

print(f"\n=== TOP 15 NO SHIFT (tickers scanned: {len(uniq)}) ===")
print(f"{'tk':<8} {'RMSE':>6} {'corr':>7} {'range':>15}  {'window':>28}")
for r in noshift[:15]:
    rmse, tk, off, corr, d0, d1, mn, mx = r
    print(f"{tk:<8} {rmse:>6.0f} {corr:>+6.3f} [{mn:>5.0f},{mx:>5.0f}]  {d0}→{d1}")

print(f"\n=== TOP 15 WITH SHIFT <=5% ===")
print(f"{'tk':<8} {'RMSE':>6} {'corr':>7} {'shift':>7}  {'window':>28}")
for r in shift5[:15]:
    rmse, tk, off, corr, shift, d0, d1 = r
    pct = 100.0 * shift / (np.median(our) - shift)
    print(f"{tk:<8} {rmse:>6.0f} {corr:>+6.3f} {shift:>+6.1f}  {d0}→{d1}  ({pct:+.2f}%)")
