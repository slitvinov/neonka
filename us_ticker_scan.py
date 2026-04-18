import os
import sys
import json
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

CACHE = Path('/tmp/yfcache')
CACHE.mkdir(exist_ok=True)

def load_facts():
    vals = []
    with open('/tmp/ticker_facts.txt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) == 2:
                vals.append(float(parts[1]))
    return np.array(vals) / 100.0

def fetch(ticker, start, end):
    fname = CACHE / f'{ticker.replace("^","_").replace("=","_")}_{start}_{end}.csv'
    if fname.exists() and fname.stat().st_size > 100:
        try:
            df = pd.read_csv(fname, index_col=0, parse_dates=True)
            if len(df) > 10:
                return df
        except Exception:
            pass
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if df is None or len(df) == 0:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.to_csv(fname)
        return df
    except Exception as e:
        print(f'  err {ticker}: {e}', file=sys.stderr)
        return None

def session_medians(df):
    return ((df['High'].values + df['Low'].values) / 2.0)

def score(target, cand):
    n = len(target)
    if len(cand) < n:
        return None
    best = None
    for off in range(0, len(cand) - n + 1):
        w = cand[off:off+n]
        if np.any(np.isnan(w)) or np.any(w <= 0):
            continue
        rmse = float(np.sqrt(np.mean((w - target) ** 2)))
        corr = float(np.corrcoef(w, target)[0, 1])
        shift = float(np.mean(w - target))
        out = {'offset': off, 'rmse': rmse, 'corr': corr, 'shift': shift}
        if best is None or rmse < best['rmse']:
            best = out
    return best

def scan(tickers, windows, target):
    results = []
    for t in tickers:
        for (ws, we) in windows:
            df = fetch(t, ws, we)
            if df is None or len(df) < len(target):
                continue
            cand = session_medians(df)
            s = score(target, cand)
            if s is None:
                continue
            s['ticker'] = t
            s['window'] = f'{ws}_{we}'
            s['dates_start'] = str(df.index[s['offset']].date())
            s['dates_end'] = str(df.index[s['offset'] + len(target) - 1].date())
            results.append(s)
    results.sort(key=lambda r: r['rmse'])
    return results

if __name__ == '__main__':
    target = load_facts()
    print(f'target: n={len(target)} min={target.min():.2f} max={target.max():.2f} drawdown={(target.min()/target.max()-1)*100:.2f}%')
    print(f'peak ses {int(np.argmax(target))} trough ses {int(np.argmin(target))}')
    for i in range(1, len(target)):
        pct = (target[i] - target[i-1]) / target[i-1] * 100
        if abs(pct) > 4:
            print(f'  big gap ses {i-1}->{i}: {pct:+.2f}%')

    tickers = ['META','NFLX','TGT','LULU','CRWD','NKE','SBUX','CSCO','PYPL','V','MA','JPM',
               'BAC','WFC','DIS','NVDA','AMD','INTC','AAPL','MSFT','GOOGL','AMZN','TSLA',
               'XOM','CVX','HD','LOW','WMT','COST','PG','KO','PEP','MCD','ORCL','IBM',
               'ADBE','CRM','NOW','SHOP','UBER','LYFT','BA','CAT','DE','GE','F','GM',
               'C','GS','MS','BLK','UNH','JNJ','PFE','MRK','LLY','ABT','TMO','ABBV',
               'SPOT','SNAP','PINS','TWTR','EBAY','ZM','DOCU','ROKU','SQ','HOOD','COIN',
               'RBLX','DASH','ABNB','MRNA','BNTX','NOK','ERIC','BABA','JD','PDD','NIO',
               'XPEV','LI','RIVN','LCID','PLTR','ASAN','U','NET','DDOG','SNOW','MDB',
               'OKTA','ZS','PANW','FTNT','ANET','AVGO','QCOM','TXN','MU','LRCX','AMAT',
               'KLAC','WDAY','TEAM','SPLK','VMW','CTSH','ACN','HPQ','DELL','LOGI',
               'ADM','MO','PM','BMY','GILD','AMGN','VRTX','ISRG','DHR','SYK','BDX',
               'ELV','CI','CVS','WBA','HUM','ANTM','HCA','UNP','CSX','NSC','FDX','UPS',
               'DAL','UAL','AAL','LUV','MAR','HLT','CCL','RCL','NCLH','MGM','LVS','WYNN',
               'ETSY','EBAY','W','CHWY','BBY','KSS','M','JWN','GPS','LB','ANF','ULTA',
               'DG','DLTR','KR','SYY','KHC','GIS','K','HSY','CPB','CL','KMB','CLX',
               'LIN','APD','ECL','SHW','NEM','FCX','BHP','RIO','VALE','MT','NUE','X',
               'DOW','DD','LYB','PXD','EOG','OXY','SLB','HAL','BKR','MPC','PSX','VLO',
               'KMI','WMB','EPD','ET','OKE','PAA','LNG','CTVA','MOS','CF','FMC',
               'CMG','QSR','DPZ','PZZA','DNUT','PEP','KDP','MNST','STZ','TAP','BUD',
               'RH','POOL','NVR','DHI','LEN','PHM','TOL','KBH','MTH','MDC','TMHC',
               'PKG','IP','WRK','SEE','AVY','BALL','OI','BRK-B','MMM','HON','ITW','ETN',
               'EMR','PH','ROK','DOV','XYL','FTV','AME','CMI','LMT','RTX','GD','NOC','LHX','TDG',
               'QQQ','SPY','IWM','DIA','VTI','XLF','XLK','XLE','XLV','XLI','XLY','XLP','XLU','XLB','XLRE','XBI','SMH','IBB','GDX']

    windows = [
        ('2019-01-01','2020-06-30'),
        ('2020-01-01','2021-03-01'),
        ('2020-06-01','2021-06-30'),
        ('2021-01-01','2022-06-30'),
        ('2022-01-01','2023-03-31'),
        ('2023-01-01','2024-03-31'),
        ('2024-01-01','2025-03-31'),
        ('2025-01-01','2026-04-18'),
    ]

    results = scan(tickers, windows, target)
    print(f'\n=== top 20 matches ===')
    print(f'{"ticker":<8}{"corr":>7}{"rmse":>8}{"shift":>8}  start         end')
    for r in results[:20]:
        print(f'{r["ticker"]:<8}{r["corr"]:>7.3f}{r["rmse"]:>8.3f}{r["shift"]:>8.3f}  {r["dates_start"]} -> {r["dates_end"]}')
