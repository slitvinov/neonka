#!/usr/bin/env python3
"""compare.py — side-by-side comparison of two LOB book histories.

Each source can be:
  data/train.raw              whole file (auto-detects sessions.raw in same dir)
  data/train.raw:3            session 3 (requires sessions.raw nearby)
  data/sim.raw:odd            odd-indexed rows  (simulated frames from simulate.c)
  data/sim.raw:even           even-indexed rows (input frames from simulate.c)

Flags override positional specs:
  -D1/-S1/-s1   source A  data/sessions/session
  -D2/-S2/-s2   source B  data/sessions/session

Adding a new comparator
-----------------------
1. Write a function  f(A, B) -> iterable of Row
2. Append Section(title, f) to SECTIONS

Row types (import from this module or create inline):
  Row(label, va, vb, fmt=".4f", diff="ratio")   scalar row: label | A | B | B/A
  DistRow(key, va, vb, diff="diff")              keyed row:  key   | A | B | diff
  Sep()                                          separator line
"""
import sys, os, argparse
from dataclasses import dataclass
from typing import Any, Iterable
import numpy as np

NL       = 8
ROW_COLS = 49
_ASK_RATE = slice(0, 8)
_BID_RATE = slice(8, 16)
_ASK_NC   = slice(32, 40)
_BID_NC   = slice(40, 48)

# ── display rows ──────────────────────────────────────────────────────────────

@dataclass
class Row:
    """Scalar comparison row: label | A | B | B/A-or-diff."""
    label: str
    a:     Any
    b:     Any
    fmt:   str  = ".4f"
    diff:  str  = "ratio"   # "ratio" | "diff" | "none"

@dataclass
class DistRow:
    """Keyed distribution row (key printed in first column)."""
    key:  Any
    a:    Any
    b:    Any
    fmt:  str = ".4f"
    diff: str = "diff"      # "ratio" | "diff" | "none"

@dataclass
class Sep:
    """Separator line."""
    char: str = "-"


# ── section registry ──────────────────────────────────────────────────────────

@dataclass
class Section:
    title:     str
    fn:        Any          # callable(A, B) -> Iterable[Row|DistRow|Sep]
    key_label: str  = ""   # label for first column in DistRow sections


SECTIONS: list[Section] = []

def section(title, *, key_label=""):
    """Decorator — registers a comparator in SECTIONS."""
    def deco(fn):
        SECTIONS.append(Section(title, fn, key_label))
        return fn
    return deco


# ── Source ────────────────────────────────────────────────────────────────────

class Source:
    def __init__(self, rows_np, label=""):
        self.rows  = rows_np
        self.label = label

    def ask_rate(self): return self.rows[:, _ASK_RATE].astype(np.int32)
    def bid_rate(self): return self.rows[:, _BID_RATE].astype(np.int32)
    def ask_nc(self):   return self.rows[:, _ASK_NC].astype(np.int32)
    def bid_nc(self):   return self.rows[:, _BID_NC].astype(np.int32)

    def spread(self):
        return self.ask_rate()[:, 0] - self.bid_rate()[:, 0]

    def nc0(self):
        return self.ask_nc()[:, 0]

    def nc1(self):
        return self.ask_nc()[:, 1]

    def mid2(self):
        return self.ask_rate()[:, 0] + self.bid_rate()[:, 0]

    def returns(self, lag=1):
        m = self.mid2()
        return (m[lag:] - m[:-lag]).astype(np.float64)

    def n_ticks(self):
        return len(self.rows)

    def events_per_pair(self):
        """Events/pair = sum |ΔN| across all 16 level-slots per adjacent pair,
        averaged. This is the event-count analog of classify_pairs.py."""
        aN = self.ask_nc(); bN = self.bid_nc()
        aR = self.ask_rate(); bR = self.bid_rate()
        n = len(self.rows) - 1
        if n <= 0: return 0.0
        cnt = 0.0
        for t in range(n):
            cnt += _walk_side_count(aR[t], aN[t], aR[t+1], aN[t+1], +1)
            cnt += _walk_side_count(bR[t], bN[t], bR[t+1], bN[t+1], -1)
        return cnt / n


def _walk_side_count(pR, pN, cR, cN, diff):
    """Elementary-event count on one side via merge-walk (mirrors rates.c/events.c)."""
    i = j = n = 0
    while i < 8 and j < 8 and pN[i] != 0 and cN[j] != 0:
        d = diff * (int(cR[j]) - int(pR[i]))
        if d < 0:
            n += int(cN[j]); j += 1
        elif d == 0:
            dn = int(cN[j]) - int(pN[i])
            if dn: n += abs(dn)
            i += 1; j += 1
        else:
            n += int(pN[i]); i += 1
    return n



# ── loading helpers ───────────────────────────────────────────────────────────

def find_sessions_file(data_path):
    d = os.path.dirname(os.path.abspath(data_path))
    for name in ("sessions.raw", "session.raw"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def load_source(data_path, sess_path=None, session_id=None, row_sel=None,
                stride=1, label=None):
    if label is None:
        label = os.path.basename(data_path)
        if session_id is not None: label += f":{session_id}"
        if row_sel: label += f":{row_sel}"
        if stride > 1: label += f":stride{stride}"

    if sess_path is None:
        sess_path = find_sessions_file(data_path)

    if not os.path.exists(data_path):
        raise FileNotFoundError(data_path)
    raw = np.fromfile(data_path, dtype=np.int32).reshape(-1, ROW_COLS)

    if session_id is not None and sess_path and os.path.exists(sess_path):
        bounds = np.fromfile(sess_path, dtype=np.int64)
        if 0 <= session_id < len(bounds) - 1:
            raw = raw[int(bounds[session_id]):int(bounds[session_id + 1])]
        else:
            print(f"warning: session {session_id} out of range", file=sys.stderr)

    if row_sel == 'even':
        raw = raw[0::2*stride]
    elif row_sel == 'odd':
        raw = raw[1::2*stride]
    elif isinstance(row_sel, int) and row_sel > 1:
        raw = raw[::row_sel]

    return Source(raw, label)


def parse_source_spec(spec):
    """Parse 'path[:tag[:stride]]'.

    tag    : 'even' | 'odd' | integer session number
    stride : integer ≥ 1  (only with even/odd; default 1)

    Examples:
      data/train.raw            → whole file
      data/train.raw:3          → session 3
      data/sim.raw:even         → even rows (actual states from onestep/simulate)
      data/sim.raw:odd          → odd  rows (predicted states)
      data/sim.raw:even:2       → every 2nd pair (skip correlated pairs)
      data/sim.raw:odd:2        → matching predicted rows
    """
    parts = spec.split(':')
    # path:even:stride  or  path:odd:stride
    if len(parts) == 3 and parts[1] in ('even', 'odd'):
        path, tag, stride_str = parts
        try:
            stride = int(stride_str)
        except ValueError:
            raise ValueError(f"stride must be an integer in '{spec}'")
        return path, None, tag, stride
    # path:tag  (tag = even/odd/session)
    if len(parts) == 2:
        path, tag = parts
        if tag in ('even', 'odd'):
            return path, None, tag, 1
        try:
            return path, int(tag), None, 1
        except ValueError:
            raise ValueError(f"unknown source tag '{tag}' in '{spec}'")
    # plain path
    if len(parts) == 1:
        return spec, None, None, 1
    raise ValueError(f"unrecognized source spec '{spec}'")


# ── display engine ────────────────────────────────────────────────────────────

W = 14   # value column width

def _fmt_val(v, fmt):
    if v is None: return "n/a"
    return format(v, fmt)

def _fmt_diff(va, vb, diff, fmt):
    if va is None or vb is None: return ""
    if diff == "ratio":
        return "" if va == 0 else f"{vb/va:.3f}"
    if diff == "diff":
        # for pct distributions show +/- points
        try:
            d = vb - va
            sign = "+" if d >= 0 else ""
            return f"{sign}{d:.2f}"
        except Exception:
            return ""
    return ""

def _print_header(sec: Section):
    print(f"\n{'━'*3} {sec.title} {'━'*(65 - len(sec.title))}")

def _render(rows: Iterable, key_label: str = ""):
    """Render rows to stdout."""
    first_dist = True
    for r in rows:
        if isinstance(r, Sep):
            print("  " + r.char * 58)
        elif isinstance(r, Row):
            if r.a is None and r.b is None:
                # sub-header / blank row — just print the label
                print(f"  {r.label}")
                continue
            sa = _fmt_val(r.a, r.fmt)
            sb = _fmt_val(r.b, r.fmt)
            sd = _fmt_diff(r.a, r.b, r.diff, r.fmt)
            print(f"  {r.label:<20} {sa:>{W}}  {sb:>{W}}  {sd:>8}")
        elif isinstance(r, DistRow):
            if first_dist:
                kl = key_label or "key"
                print(f"  {kl:>6}  {'A':>{W}}  {'B':>{W}}  {'diff/ratio':>10}")
                print("  " + "-" * 58)
                first_dist = False
            sa = _fmt_val(r.a, r.fmt)
            sb = _fmt_val(r.b, r.fmt)
            sd = _fmt_diff(r.a, r.b, r.diff, r.fmt)
            print(f"  {r.key:>6}  {sa:>{W}}  {sb:>{W}}  {sd:>10}")


def run_sections(A, B, sections):
    print(f"\n{'═'*70}")
    print(f"  A: {A.label}")
    print(f"  B: {B.label}")
    print(f"{'═'*70}")
    print(f"  {'':20} {'A':>{W}}  {'B':>{W}}  {'B/A':>8}")

    for sec in sections:
        _print_header(sec)
        try:
            rows = list(sec.fn(A, B))
        except Exception as e:
            print(f"  (error: {e})")
            continue
        _render(rows, sec.key_label)

    print(f"\n{'═'*70}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Comparators — add new ones here
# ══════════════════════════════════════════════════════════════════════════════

@section("0. FLAGS  (sim-vs-real headline gaps — surfaces big regressions)")
def flags(A, B):
    rows = []
    evA, evB = A.events_per_pair(), B.events_per_pair()
    if evA > 0 and abs(evB - evA) / evA > 0.05:
        rows.append(Row(f"⚠ events/pair off by {100*(evB/evA - 1):+.1f}%",
                        evA, evB, fmt=".4f", diff="ratio"))
    spA, spB = A.spread().mean(), B.spread().mean()
    if spA > 0 and abs(spB - spA) / spA > 0.05:
        rows.append(Row(f"⚠ spread mean off by {100*(spB/spA - 1):+.1f}%",
                        spA, spB, fmt=".3f", diff="ratio"))
    nA, nB = A.nc0().mean(), B.nc0().mean()
    if nA > 0 and abs(nB - nA) / nA > 0.05:
        rows.append(Row(f"⚠ nc0 mean off by {100*(nB/nA - 1):+.1f}%",
                        nA, nB, fmt=".3f", diff="ratio"))
    n1A, n1B = A.nc1().mean(), B.nc1().mean()
    if n1A > 0 and abs(n1B - n1A) / n1A > 0.05:
        rows.append(Row(f"⚠ nc1 mean off by {100*(n1B/n1A - 1):+.1f}%",
                        n1A, n1B, fmt=".3f", diff="ratio"))
    retA, retB = A.returns(1).std(), B.returns(1).std()
    if retA > 0 and abs(retB - retA) / retA > 0.05:
        rows.append(Row(f"⚠ return std off by {100*(retB/retA - 1):+.1f}%",
                        retA, retB, fmt=".3f", diff="ratio"))
    if not rows:
        rows.append(Row("no gross discrepancies (<5% on ev rate / spread / nc0 / ret std)",
                        1.0, 1.0, fmt=".1f", diff="none"))
    return rows


@section("1. BASICS")
def basics(A, B):
    sp_a, sp_b = A.spread(), B.spread()
    return [
        Row("ticks",            A.n_ticks(),         B.n_ticks(),         fmt=",d", diff="none"),
        Row("events/pair",      A.events_per_pair(), B.events_per_pair(), fmt=".4f", diff="ratio"),
        Row("spread mean",      sp_a.mean(),         sp_b.mean()),
        Row("spread std",       sp_a.std(),          sp_b.std()),
        Row("spread median",    np.median(sp_a),     np.median(sp_b),     fmt=".1f"),
        Row("nc0 mean",         A.nc0().mean(),      B.nc0().mean()),
        Row("nc0 std",          A.nc0().std(),       B.nc0().std()),
        Row("nc1 mean",         A.nc1().mean(),      B.nc1().mean()),
        Row("nc1 std",          A.nc1().std(),       B.nc1().std()),
    ]


@section("2. SPREAD DISTRIBUTION  (% of ticks)", key_label="sp0")
def spread_distribution(A, B):
    def counts(src):
        sp = src.spread()
        v, c = np.unique(sp, return_counts=True)
        return dict(zip(v.tolist(), (100.0 * c / len(sp)).tolist()))

    da, db = counts(A), counts(B)
    all_vals = sorted(set(da) | set(db))
    rows = [Sep()]
    for v in all_vals:
        pa, pb = da.get(v, 0.0), db.get(v, 0.0)
        if pa < 0.5 and pb < 0.5: continue
        rows.append(DistRow(v, pa, pb, fmt=".2f", diff="diff"))
    return rows


@section("3. NC0 DISTRIBUTION  (% of ticks, ask side)", key_label="nc0")
def nc0_distribution(A, B):
    def counts(src):
        nc = np.minimum(src.nc0(), 10)
        return {v: 100.0 * (nc == v).mean() for v in range(1, 11)}

    da, db = counts(A), counts(B)
    rows = [Sep()]
    for v in range(1, 11):
        pa, pb = da[v], db[v]
        if pa < 0.5 and pb < 0.5: continue
        rows.append(DistRow(v, pa, pb, fmt=".2f", diff="diff"))
    return rows


@section("4. BOOK DEPTH PROFILE  (ask, when level present)")
def depth_profile(A, B):
    def profile(src):
        ar = src.ask_rate(); an = src.ask_nc()
        present = an > 0
        avg_n = [an[:, i][present[:, i]].mean() if present[:, i].any() else 0.0
                 for i in range(NL)]
        d = (ar - ar[:, 0:1]) * 2
        avg_d = [d[:, i][present[:, i]].mean() if present[:, i].any() else float('nan')
                 for i in range(NL)]
        frac  = [present[:, i].mean() for i in range(NL)]
        return avg_n, avg_d, frac

    na, da, fa = profile(A)
    nb, db, fb = profile(B)
    print(f"  {'lvl':>4}  {'dist_A':>{W}}  {'dist_B':>{W}}  "
          f"{'avgN_A':>8}  {'avgN_B':>8}  {'frac_A':>7}  {'frac_B':>7}")
    print("  " + "-" * 72)
    for i in range(NL):
        if fa[i] < 0.01 and fb[i] < 0.01: continue
        print(f"  {i:>4}  {da[i]:{W}.1f}  {db[i]:{W}.1f}  "
              f"{na[i]:>8.2f}  {nb[i]:>8.2f}  "
              f"{100*fa[i]:>6.1f}%  {100*fb[i]:>6.1f}%")
    return []   # already printed directly


@section("5. PRICE-RETURN DISTRIBUTION  (lag-1, half-ticks)")
def return_distribution(A, B):
    try:
        from scipy import stats as sp_stats
        def stats(src):
            r = src.returns(1)
            return (float(r.mean()), float(r.std()),
                    float(sp_stats.skew(r)), float(sp_stats.kurtosis(r)))
        ma, sa, ska, ka = stats(A)
        mb, sb, skb, kb = stats(B)
        return [
            Row("mean",          ma,  mb),
            Row("std",           sa,  sb),
            Row("skewness",      ska, skb, diff="none"),
            Row("exc. kurtosis", ka,  kb,  diff="none"),
        ]
    except ImportError:
        ra, rb = A.returns(1), B.returns(1)
        return [Row("mean", float(ra.mean()), float(rb.mean())),
                Row("std",  float(ra.std()),  float(rb.std()))]


@section("6. PRICE-RETURN AUTOCORRELATION", key_label="lag")
def price_autocorrelation(A, B):
    lags = (1, 2, 3, 5, 10, 20, 50)

    def acf(src, lags):
        r = src.returns(1)
        if len(r) < max(lags) + 10: return [None] * len(lags)
        r0 = r - r.mean(); var = float((r0 ** 2).mean())
        if var == 0: return [0.0] * len(lags)
        return [float((r0[:-k] * r0[k:]).mean()) / var for k in lags]

    acf_a, acf_b = acf(A, lags), acf(B, lags)
    rows = [Sep()]
    for lag, va, vb in zip(lags, acf_a, acf_b):
        rows.append(DistRow(lag, va, vb, fmt=".4f", diff="none"))
    return rows


@section("7. MID-PRICE PREDICTION  (B.mid - A.mid per pair)")
def price_prediction(A, B):
    ma = A.mid2().astype(np.float64) / 2.0
    mb = B.mid2().astype(np.float64) / 2.0
    n = min(len(ma), len(mb))
    if n == 0:
        return []
    d = mb[:n] - ma[:n]
    y = A.rows[:n, 48].astype(np.float64) / 4.0
    dc = d - d.mean()
    yc = y - y.mean()
    denom = np.sqrt((dc*dc).sum() * (yc*yc).sum())
    corr = float((dc*yc).sum() / denom) if denom > 0 else 0.0
    sstot = float((yc*yc).sum())
    if sstot > 0 and (dc*dc).sum() > 0:
        beta  = float((dc*yc).sum() / (dc*dc).sum())
        ssres = float(((y - beta*d)**2).sum())
        r2_fit = 1.0 - ssres / sstot
    else:
        r2_fit = 0.0
    return [
        Row("pairs",         n,                      n,                  diff="none"),
        Row("dmid mean",     float(d.mean()),        None,               diff="none"),
        Row("dmid std",      float(d.std()),         None,               diff="none"),
        Row("|dmid| mean",   float(np.abs(d).mean()),None,               diff="none"),
        Row("dmid > 0 %",    float(100*(d > 0).mean()),   None,          diff="none"),
        Row("dmid = 0 %",    float(100*(d == 0).mean()),  None,          diff="none"),
        Row("dmid < 0 %",    float(100*(d < 0).mean()),   None,          diff="none"),
        Row("corr(dmid, y)", corr,                   None,               diff="none"),
        Row("R^2 = corr^2",  corr*corr,              None,               diff="none"),
        Row("R^2 (best-fit)", r2_fit,                None,               diff="none"),
    ]


@section("12. RETURN VOL BY SPREAD  (std of lag-1 return, per sp0)", key_label="sp0")
def return_vol_by_spread(A, B):
    def vol_by_sp(src):
        sp  = src.spread()[:-1]          # sp at t
        ret = src.returns(1)             # mid change from t to t+1
        result = {}
        for v in np.unique(sp):
            mask = sp == v
            if mask.sum() < 50: continue
            result[int(v)] = float(ret[mask].std())
        return result

    va, vb = vol_by_sp(A), vol_by_sp(B)
    all_vals = sorted(set(va) | set(vb))
    rows = [Sep()]
    for v in all_vals:
        pa, pb = va.get(v), vb.get(v)
        if pa is None and pb is None: continue
        if (pa or 0) < 1e-6 and (pb or 0) < 1e-6: continue
        rows.append(DistRow(v, pa, pb, fmt=".4f", diff="ratio"))
    return rows


@section("13. QUEUE IMBALANCE  (ask_nc0 - bid_nc0) / (ask_nc0 + bid_nc0)")
def queue_imbalance(A, B):
    def imbalance(src):
        ask = src.ask_nc()[:, 0].astype(float)
        bid = src.bid_nc()[:, 0].astype(float)
        total = ask + bid
        imb = np.where(total > 0, (ask - bid) / total, 0.0)
        return imb

    ia, ib = imbalance(A), imbalance(B)
    rows = [
        Row("mean",     float(ia.mean()),             float(ib.mean())),
        Row("std",      float(ia.std()),               float(ib.std())),
        Row("|mean|",   float(np.abs(ia).mean()),      float(np.abs(ib).mean())),
        Row("ask-heavy% (>0.2)",
            float((ia >  0.2).mean() * 100),
            float((ib >  0.2).mean() * 100), fmt=".2f", diff="ratio"),
        Row("bid-heavy% (>0.2)",
            float((ia < -0.2).mean() * 100),
            float((ib < -0.2).mean() * 100), fmt=".2f", diff="ratio"),
        Row("balanced%  (|imb|<0.1)",
            float((np.abs(ia) < 0.1).mean() * 100),
            float((np.abs(ib) < 0.1).mean() * 100), fmt=".2f", diff="ratio"),
    ]
    return rows


IMB_EDGES  = [-1.001, -0.3, -0.1, 0.1, 0.3, 1.001]
IMB_LABELS = ["<-.3", "-.3..-.1", "-.1..+.1", "+.1..+.3", ">+.3"]

def _imb0(src):
    a = src.ask_nc()[:, 0].astype(float)
    b = src.bid_nc()[:, 0].astype(float)
    return np.where(a + b > 0, (a - b) / (a + b), 0.0)

def _bin_imb(imb):
    return np.digitize(imb, IMB_EDGES) - 1


@section("14. E[Δmid | imb_bin]  (directional drift per state)", key_label="imb")
def drift_by_imb(A, B):
    def drift(src):
        imb = _imb0(src)[:-1]
        dm  = src.returns(1)
        bins = _bin_imb(imb)
        out = {}
        for i, lbl in enumerate(IMB_LABELS):
            m = bins == i
            out[lbl] = float(dm[m].mean()) if m.sum() > 50 else None
        return out
    da, db = drift(A), drift(B)
    rows = [Sep()]
    for lbl in IMB_LABELS:
        rows.append(DistRow(lbl, da.get(lbl), db.get(lbl), fmt="+.4f", diff="diff"))
    return rows


@section("15. P(best-quote move) | imb_bin  (%)", key_label="imb")
def quote_moves_by_imb(A, B):
    def pmoves(src):
        aR, bR = src.ask_rate()[:, 0], src.bid_rate()[:, 0]
        imb = _imb0(src)[:-1]
        bins = _bin_imb(imb)
        out = {}
        for i, lbl in enumerate(IMB_LABELS):
            m = bins == i
            if m.sum() < 50:
                out[lbl] = None; continue
            aU = (aR[1:][m] > aR[:-1][m]).mean() * 100
            aD = (aR[1:][m] < aR[:-1][m]).mean() * 100
            bU = (bR[1:][m] > bR[:-1][m]).mean() * 100
            bD = (bR[1:][m] < bR[:-1][m]).mean() * 100
            out[lbl] = (aU, aD, bU, bD)
        return out
    da, db = pmoves(A), pmoves(B)
    print(f"  {'imb':>8}  {'ask↑_A':>7} {'ask↑_B':>7}   {'ask↓_A':>7} {'ask↓_B':>7}"
          f"   {'bid↑_A':>7} {'bid↑_B':>7}   {'bid↓_A':>7} {'bid↓_B':>7}")
    print("  " + "-" * 88)
    for lbl in IMB_LABELS:
        va, vb = da.get(lbl), db.get(lbl)
        if va is None or vb is None: continue
        print(f"  {lbl:>8}  "
              f"{va[0]:>6.2f}% {vb[0]:>6.2f}%   {va[1]:>6.2f}% {vb[1]:>6.2f}%"
              f"   {va[2]:>6.2f}% {vb[2]:>6.2f}%   {va[3]:>6.2f}% {vb[3]:>6.2f}%")
    return []


@section("16. P(side event) | imb_bin  (% of pairs with a change on that side)", key_label="imb")
def side_event_by_imb(A, B):
    def pside(src):
        R = src.rows
        imb = _imb0(src)[:-1]
        a_chg = (np.any(R[1:, 0:8] != R[:-1, 0:8], axis=1)
               | np.any(R[1:, 16:24] != R[:-1, 16:24], axis=1)
               | np.any(R[1:, 32:40] != R[:-1, 32:40], axis=1))
        b_chg = (np.any(R[1:, 8:16] != R[:-1, 8:16], axis=1)
               | np.any(R[1:, 24:32] != R[:-1, 24:32], axis=1)
               | np.any(R[1:, 40:48] != R[:-1, 40:48], axis=1))
        bins = _bin_imb(imb)
        out = {}
        for i, lbl in enumerate(IMB_LABELS):
            m = bins == i
            if m.sum() < 50:
                out[lbl] = None; continue
            out[lbl] = (100 * a_chg[m].mean(), 100 * b_chg[m].mean())
        return out
    da, db = pside(A), pside(B)
    print(f"  {'imb':>8}  {'ask_A':>6} {'ask_B':>6}   {'bid_A':>6} {'bid_B':>6}   {'A asym':>7} {'B asym':>7}")
    print("  " + "-" * 72)
    for lbl in IMB_LABELS:
        va, vb = da.get(lbl), db.get(lbl)
        if va is None or vb is None: continue
        asy_a = va[0] - va[1]
        asy_b = vb[0] - vb[1]
        print(f"  {lbl:>8}  {va[0]:>5.2f}% {vb[0]:>5.2f}%   {va[1]:>5.2f}% {vb[1]:>5.2f}%"
              f"   {asy_a:>+6.2f} {asy_b:>+6.2f}")
    return []


@section("17. ACTIVITY-RATE ACF  (rate clustering — Poisson sim should give ~0)", key_label="lag")
def activity_acf(A, B):
    lags = (1, 5, 20, 50, 200, 500)
    def act_acf(src):
        R = src.rows
        chg = np.any(R[1:, :48] != R[:-1, :48], axis=1).astype(float)
        W = 50
        if len(chg) < W + max(lags) + 10:
            return [None] * len(lags)
        rate = np.convolve(chg, np.ones(W)/W, mode='valid')
        r0 = rate - rate.mean(); v = (r0*r0).mean()
        if v == 0: return [0.0] * len(lags)
        return [float((r0[:-k] * r0[k:]).mean() / v) for k in lags]
    aa, ab = act_acf(A), act_acf(B)
    rows = [Sep()]
    for lag, va, vb in zip(lags, aa, ab):
        rows.append(DistRow(lag, va, vb, fmt=".4f", diff="diff"))
    return rows


@section("18. E[Δmid | sp_bin × imb_bin]  (2D drift grid — data shows sweep, Poisson sim flat)")
def drift_grid(A, B):
    def grid(src):
        sp  = src.spread()[:-1]
        imb = _imb0(src)[:-1]
        dm  = src.returns(1)
        bins = _bin_imb(imb)
        sp_vals = [2, 4, 6, 8, 10, 12]
        out = {}
        for v in sp_vals:
            row = []
            for i in range(len(IMB_LABELS)):
                m = (sp == v) & (bins == i)
                row.append(float(dm[m].mean()) if m.sum() > 100 else None)
            out[v] = row
        return out
    ga, gb = grid(A), grid(B)
    hdr = "  " + f"{'sp':>3}  " + "  ".join(f"{lbl:>10}" for lbl in IMB_LABELS)
    print(hdr + "  (A=data, B=sim)")
    print("  " + "-" * (len(hdr) - 2))
    for v in sorted(ga):
        row_a = "  ".join(f"{x:>+10.4f}" if x is not None else f"{'—':>10}" for x in ga[v])
        row_b = "  ".join(f"{x:>+10.4f}" if x is not None else f"{'—':>10}" for x in gb[v])
        print(f"  A{v:>2}  {row_a}")
        print(f"  B{v:>2}  {row_b}")
    return []


@section("19. |Δmid| TAIL  (heavy tails = sim blow-ups)", key_label="pctl")
def dmid_tails(A, B):
    def tails(src):
        r = np.abs(src.returns(1))
        return {p: float(np.percentile(r, p)) for p in (50, 75, 90, 95, 99, 99.5, 99.9, 100)}
    ta, tb = tails(A), tails(B)
    rows = [Sep()]
    for p in (50, 75, 90, 95, 99, 99.5, 99.9, 100):
        lbl = "max" if p == 100 else f"p{p}"
        rows.append(DistRow(lbl, ta[p], tb[p], fmt=".3f", diff="ratio"))
    return rows


@section("20. BOOK DEPTH TAIL  (ask_dist[7]: extreme deep-level behavior)", key_label="pctl")
def depth_tail(A, B):
    def pctl(src):
        ar = src.ask_rate()
        an = src.ask_nc()
        present = an[:, 7] > 0
        if present.sum() < 50: return {p: None for p in (50, 90, 99, 99.9)}
        d = 2 * (ar[:, 7][present] - ar[:, 0][present])
        return {p: float(np.percentile(d, p)) for p in (50, 90, 99, 99.9)}
    ta, tb = pctl(A), pctl(B)
    rows = [Sep()]
    for p in (50, 90, 99, 99.9):
        rows.append(DistRow(f"p{p}", ta[p], tb[p], fmt=".1f", diff="diff"))
    return rows


@section("21. IMBALANCE TRANSITIONS  (regime-flipping frequency)", key_label="metric")
def imb_transitions(A, B):
    def metrics(src):
        imb = _imb0(src)
        # |imb| exceedance
        ex50 = float((np.abs(imb) > 0.5).mean() * 100)
        ex30 = float((np.abs(imb) > 0.3).mean() * 100)
        # transition: cross ±0.2 from opposite side
        sign_bin = np.where(imb > 0.2, 1, np.where(imb < -0.2, -1, 0))
        flips = 0; last = 0
        for v in sign_bin:
            if v != 0 and v != last and last != 0:
                flips += 1
            if v != 0: last = v
        flip_per_1k = 1000.0 * flips / max(1, len(imb))
        return {"|imb|>0.5 %": ex50, "|imb|>0.3 %": ex30, "flips /1000 ticks": flip_per_1k}
    ma, mb = metrics(A), metrics(B)
    rows = [Sep()]
    for k in ma:
        rows.append(DistRow(k, ma[k], mb[k], fmt=".3f", diff="ratio"))
    return rows


@section("22. LAG ASYMMETRY — corr(event_side_marker, Δmid[t+k])", key_label="lag k")
def lag_asymmetry(A, B):
    def lags(src):
        R = src.rows
        mid = (R[:, 0].astype(np.float64) + R[:, 8]) / 2.0
        dm  = np.diff(mid)
        aN0 = R[:-1, 32].astype(float); bN0 = R[:-1, 40].astype(float)
        sig = (aN0 - bN0) / np.maximum(aN0 + bN0, 1)
        out = {}
        for k in (-5, -2, -1, 0, 1, 2, 5):
            if k < 0:
                x = sig[-k:]; y = dm[:k]
            elif k == 0:
                x = sig; y = dm
            else:
                x = sig[:-k]; y = dm[k:]
            nmin = min(len(x), len(y))
            if nmin < 100:
                out[k] = None; continue
            x, y = x[:nmin], y[:nmin]
            xc = x - x.mean(); yc = y - y.mean()
            den = np.sqrt((xc*xc).sum() * (yc*yc).sum())
            out[k] = float((xc*yc).sum()/den) if den > 0 else 0.0
        return out
    la, lb = lags(A), lags(B)
    rows = [Sep()]
    for k in (-5, -2, -1, 0, 1, 2, 5):
        rows.append(DistRow(k, la.get(k), lb.get(k), fmt="+.4f", diff="diff"))
    return rows


@section("23. tp/(tp+tm) vs dp/(dp+dm) drift WITHIN stream (10 deciles)", key_label="decile")
def flow_ratio_drift(A, B):
    # Quick approximation using book-state deltas at best (tp/tm) vs elsewhere (dp/dm)
    def drift(src):
        R = src.rows
        if len(R) < 1000: return None
        aN = R[:, 32:40]; bN = R[:, 40:48]
        # tp/tm proxies: Δ(N at best) summed, split by sign
        daN0 = np.diff(aN[:, 0]); dbN0 = np.diff(bN[:, 0])
        tp_cnt = np.clip(daN0, 0, None) + np.clip(dbN0, 0, None)   # positive deltas at best
        tm_cnt = np.clip(-daN0, 0, None) + np.clip(-dbN0, 0, None) # negative deltas at best
        # dp/dm proxies: Δ(total ask N, levels 1..7) summed
        daN_deep = np.diff(aN[:, 1:].sum(axis=1))
        dbN_deep = np.diff(bN[:, 1:].sum(axis=1))
        dp_cnt = np.clip(daN_deep, 0, None) + np.clip(dbN_deep, 0, None)
        dm_cnt = np.clip(-daN_deep, 0, None) + np.clip(-dbN_deep, 0, None)
        n = len(tp_cnt)
        W = 10
        e = np.linspace(0, n, W+1).astype(int)
        out = []
        for i in range(W):
            lo, hi = e[i], e[i+1]
            tps = tp_cnt[lo:hi].sum(); tms = tm_cnt[lo:hi].sum()
            dps = dp_cnt[lo:hi].sum(); dms = dm_cnt[lo:hi].sum()
            tpf = tps/(tps+tms) if tps+tms>0 else None
            dpf = dps/(dps+dms) if dps+dms>0 else None
            out.append((tpf, dpf))
        return out
    a = drift(A); b = drift(B)
    if a is None or b is None: return []
    rows = [Sep()]
    for k in range(10):
        ta, da = a[k]
        tb, db = b[k]
        sa = (ta + da) if (ta is not None and da is not None) else None
        sb = (tb + db) if (tb is not None and db is not None) else None
        rows.append(DistRow(f"d{k}:sum", sa, sb, fmt=".3f", diff="diff"))
    return rows


@section("24. SIM FLAG DISTRIBUTION  (y column used as flag by onestep on sim rows)")
def sim_flag_dist(A, B):
    def flags(src):
        y = src.rows[:, 48]
        total = len(y)
        if total == 0: return None
        out = {}
        for v in (0, 1, 2):
            out[f"flag {v}"] = 100.0 * (y == v).sum() / total
        out["flag other"] = 100.0 * (~np.isin(y, [0, 1, 2])).sum() / total
        return out
    fa, fb = flags(A), flags(B)
    if fa is None or fb is None: return []
    rows = []
    for k in fa:
        rows.append(Row(k, fa[k], fb[k], fmt=".3f", diff="diff"))
    return rows


@section("25. EVENT-TYPE TRANSITION  P(next | last): Markov memory check", key_label="from→to")
def event_transitions(A, B):
    def get_transitions(src):
        R = src.rows
        if len(R) < 2: return None
        aR0 = R[:-1, 0]; bR0 = R[:-1, 8]
        aR1 = R[1:, 0];  bR1 = R[1:, 8]
        aN0 = R[:-1, 32]; aN1 = R[1:, 32]
        bN0 = R[:-1, 40]; bN1 = R[1:, 40]
        ev = np.zeros(len(R)-1, dtype=int)
        # 0=none, 1=ask_add, 2=ask_rem, 3=bid_add, 4=bid_rem
        ev[(aR1==aR0) & (aN1>aN0)] = 1
        ev[((aR1!=aR0) | (aN1<aN0)) & ((aR1!=aR0) | (aN1!=aN0))] = 2
        ev[(bR1==bR0) & (bN1>bN0)] = 3
        ev[((bR1!=bR0) | (bN1<bN0)) & (bR1!=bR0) & (ev==0)] = 4
        cnt = np.zeros((5,5), dtype=np.int64)
        for k in range(1, len(ev)):
            cnt[ev[k-1], ev[k]] += 1
        row_sum = cnt.sum(axis=1)
        P = cnt / np.maximum(row_sum[:, None], 1)
        return P
    Pa = get_transitions(A); Pb = get_transitions(B)
    if Pa is None or Pb is None: return []
    labels = ["none", "a+", "a-", "b+", "b-"]
    print(f"  data A (rows), sim B (rows below)")
    print(f"  {'from':>8} → " + "  ".join(f"{l:>5}" for l in labels))
    for i, lab in enumerate(labels):
        row_a = "  ".join(f"{100*Pa[i,j]:>5.1f}" for j in range(5))
        row_b = "  ".join(f"{100*Pb[i,j]:>5.1f}" for j in range(5))
        print(f"  A {lab:>6} : {row_a}")
        print(f"  B {lab:>6} : {row_b}")
    return []


@section("26. INTER-EVENT TIME SURVIVAL  P(gap≥k) — Poisson=exp, Hawkes=heavier", key_label="k ticks")
def inter_event_survival(A, B):
    def gaps(src):
        R = src.rows
        chg = np.any(R[1:, :48] != R[:-1, :48], axis=1)
        ev_idx = np.where(chg)[0]
        if len(ev_idx) < 100: return None
        return np.diff(ev_idx)
    ga, gb = gaps(A), gaps(B)
    if ga is None or gb is None: return []
    rows = [Sep()]
    for k in [1, 2, 3, 5, 10, 20, 50, 100, 200]:
        pa = float((ga >= k).mean()); pb = float((gb >= k).mean())
        rows.append(DistRow(k, pa, pb, fmt=".4f", diff="ratio"))
    return rows


@section("27. REALIZED VOL ACF  (std of Δmid over W=50; regime persistence)", key_label="lag")
def rvol_acf(A, B):
    lags = (1, 10, 50, 200, 1000)
    def acf(src):
        R = src.rows
        if len(R) < 2000: return [None]*len(lags)
        mid = (R[:, 0].astype(np.float64) + R[:, 8]) / 2.0
        dm = np.diff(mid); W = 50
        if len(dm) < W + max(lags) + 100: return [None]*len(lags)
        rv = np.array([np.std(dm[i:i+W]) for i in range(0, len(dm)-W, 10)])
        r0 = rv - rv.mean(); v = (r0*r0).mean()
        if v == 0: return [0.0]*len(lags)
        out = []
        for lag in lags:
            lag_s = max(1, lag // 10)
            if lag_s < len(r0):
                out.append(float((r0[:-lag_s]*r0[lag_s:]).mean()/v))
            else: out.append(None)
        return out
    aa, ab = acf(A), acf(B)
    rows = [Sep()]
    for lag, va, vb in zip(lags, aa, ab):
        rows.append(DistRow(lag, va, vb, fmt=".4f", diff="diff"))
    return rows


@section("28. SPREAD RELAXATION  half-life: |sp(t) − sp_mean| → decay", key_label="lag")
def spread_relax(A, B):
    lags = (1, 5, 20, 50, 200, 1000)
    def acf(src):
        sp = src.spread().astype(float)
        if len(sp) < max(lags) + 100: return [None]*len(lags)
        s0 = sp - sp.mean(); v = (s0*s0).mean()
        if v == 0: return [0.0]*len(lags)
        return [float((s0[:-k]*s0[k:]).mean()/v) for k in lags]
    aa, ab = acf(A), acf(B)
    rows = [Sep()]
    for lag, va, vb in zip(lags, aa, ab):
        rows.append(DistRow(lag, va, vb, fmt=".4f", diff="diff"))
    return rows


@section("29. PRICE IMPACT  E[Δmid(t+k) | ask-event at t] (half-ticks)", key_label="k lag")
def price_impact(A, B):
    def impact(src):
        R = src.rows
        if len(R) < 1000: return {k: None for k in (1, 2, 5, 10, 20)}
        aR = R[:, 0]
        aN = R[:, 32]
        mid = (R[:, 0].astype(np.float64) + R[:, 8]) / 2.0
        ask_rem = (aR[1:] > aR[:-1]) | ((aR[1:] == aR[:-1]) & (aN[1:] < aN[:-1]))
        out = {}
        for k in (1, 2, 5, 10, 20):
            if len(mid) <= k + 1: out[k] = None; continue
            dm_future = mid[k+1:] - mid[1:-k]
            trig = ask_rem[:-k] if k > 0 else ask_rem
            n = min(len(dm_future), len(trig))
            out[k] = float(dm_future[:n][trig[:n]].mean()) if trig[:n].sum() > 50 else None
        return out
    ia, ib = impact(A), impact(B)
    rows = [Sep()]
    for k in (1, 2, 5, 10, 20):
        rows.append(DistRow(k, ia.get(k), ib.get(k), fmt="+.4f", diff="diff"))
    return rows


@section("30. QUEUE SIZE STATIONARY π(aN[0]) at sp=6 (% of ticks)", key_label="aN[0]")
def queue_stationary(A, B):
    def hist(src):
        sp = src.spread()
        aN0 = src.ask_nc()[:, 0]
        m = sp == 6
        if m.sum() < 500: return None
        out = {}
        for v in range(1, 11):
            out[v] = 100.0 * float((aN0[m] == v).mean())
        out["≥11"] = 100.0 * float((aN0[m] >= 11).mean())
        return out
    ha, hb = hist(A), hist(B)
    if ha is None or hb is None: return []
    rows = [Sep()]
    for k in list(range(1, 11)) + ["≥11"]:
        va = ha.get(k); vb = hb.get(k)
        if va is None and vb is None: continue
        if (va or 0) < 0.5 and (vb or 0) < 0.5: continue
        rows.append(DistRow(k, va, vb, fmt=".2f", diff="diff"))
    return rows


@section("31. MICROPRICE PREDICTIVE  corr(micro−mid, Δmid(t+k)) — single-feature check", key_label="k lag")
def micro_predictive(A, B):
    def corr_lag(src):
        R = src.rows
        if len(R) < 1000: return {k: None for k in (1, 5, 10, 50)}
        aN0 = R[:, 32].astype(np.float64); bN0 = R[:, 40].astype(np.float64)
        sp = (R[:, 0] - R[:, 8]).astype(np.float64)
        imb = (aN0 - bN0) / np.maximum(aN0 + bN0, 1)
        micro_adj = -sp * imb / 2.0
        mid = (R[:, 0].astype(np.float64) + R[:, 8]) / 2.0
        out = {}
        for k in (1, 5, 10, 50):
            if len(mid) <= k + 1: out[k] = None; continue
            y = mid[k:] - mid[:-k]
            x = micro_adj[:-k]
            n = min(len(x), len(y))
            xc = x[:n] - x[:n].mean(); yc = y[:n] - y[:n].mean()
            den = np.sqrt((xc*xc).sum() * (yc*yc).sum())
            out[k] = float((xc*yc).sum() / den) if den > 0 else 0.0
        return out
    ca, cb = corr_lag(A), corr_lag(B)
    rows = [Sep()]
    for k in (1, 5, 10, 50):
        rows.append(DistRow(k, ca.get(k), cb.get(k), fmt="+.4f", diff="diff"))
    return rows


@section("32. CONDITIONAL EVENT BURST  P(event in next k | event now) − P(event in next k | none now)", key_label="k lag")
def event_burst(A, B):
    def burst(src):
        R = src.rows
        if len(R) < 2000: return {k: None for k in (1, 2, 5, 10, 20, 50)}
        chg = (np.abs(np.diff(R[:, :48], axis=0)).sum(axis=1) > 0).astype(np.float32)
        out = {}
        for k in (1, 2, 5, 10, 20, 50):
            if len(chg) <= k + 1: out[k] = None; continue
            now_ev = chg[:-k].astype(bool)
            future = chg[k:].astype(bool)
            n = min(len(now_ev), len(future))
            if now_ev[:n].sum() < 50 or (~now_ev[:n]).sum() < 50:
                out[k] = None; continue
            p_given_ev   = float(future[:n][now_ev[:n]].mean())
            p_given_none = float(future[:n][~now_ev[:n]].mean())
            out[k] = p_given_ev - p_given_none
        return out
    ba, bb = burst(A), burst(B)
    rows = [Sep()]
    for k in (1, 2, 5, 10, 20, 50):
        rows.append(DistRow(k, ba.get(k), bb.get(k), fmt="+.4f", diff="diff"))
    return rows


@section("33. RATE SANITY  events per pair — catches calibration-scale bugs")
def rate_sanity(A, B):
    def stats(src):
        R = src.rows
        if len(R) < 100: return None
        d = np.diff(R[:, :48], axis=0)
        # any change
        any_chg = (np.abs(d).sum(axis=1) > 0)
        # ask/bid change separately
        ask_chg = ((np.abs(d[:, 0:8]).sum(axis=1)) + (np.abs(d[:, 32:40]).sum(axis=1))) > 0
        bid_chg = ((np.abs(d[:, 8:16]).sum(axis=1)) + (np.abs(d[:, 40:48]).sum(axis=1))) > 0
        # |Δ N_total|  = sum of abs(count deltas) across all levels, both sides
        dN_tot = np.abs(d[:, 32:40]).sum(axis=1) + np.abs(d[:, 40:48]).sum(axis=1)
        # level count changes
        n_levels_changed = (d[:, 32:40] != 0).sum(axis=1) + (d[:, 40:48] != 0).sum(axis=1)
        return {
            "P(≥1 event/pair)":      float(any_chg.mean()),
            "P(ask event/pair)":     float(ask_chg.mean()),
            "P(bid event/pair)":     float(bid_chg.mean()),
            "mean |ΔN_tot|/pair":    float(dN_tot.mean()),
            "mean levels chgd/pair": float(n_levels_changed.mean()),
            "P(multi-lvl chg)":      float((n_levels_changed >= 2).mean()),
            "max |ΔN_tot|":          float(dN_tot.max()),
        }
    sa, sb = stats(A), stats(B)
    if sa is None or sb is None: return []
    rows = []
    for k in sa:
        rows.append(Row(k, sa[k], sb[k], fmt=".4f", diff="ratio"))
    return rows


@section("34. PER-SIDE EVENT RATE vs spread  events/pair | sp — calibration check", key_label="sp")
def rate_vs_spread(A, B):
    def by_sp(src):
        R = src.rows
        if len(R) < 1000: return None
        sp = (R[:-1, 0] - R[:-1, 8])
        d = np.diff(R[:, :48], axis=0)
        any_chg = (np.abs(d).sum(axis=1) > 0).astype(float)
        out = {}
        for v in (2, 4, 6, 8, 10, 12, 16, 20):
            m = (sp == v)
            if m.sum() < 100: continue
            out[v] = float(any_chg[m].mean())
        return out
    ba, bb = by_sp(A), by_sp(B)
    if ba is None or bb is None: return []
    rows = [Sep()]
    for v in sorted(set(ba.keys()) & set(bb.keys())):
        rows.append(DistRow(v, ba[v], bb[v], fmt=".4f", diff="ratio"))
    return rows


if __name__ == "__main__":
    specs = sys.argv[1:]
    if len(specs) != 2:
        print("usage: compare.py <srcA> <srcB>", file=sys.stderr)
        sys.exit(1)
    srcs = []
    for spec in specs:
        path, ses, sel, stride = parse_source_spec(spec)
        srcs.append(load_source(path, None, ses, sel, stride))
    run_sections(srcs[0], srcs[1], SECTIONS)


