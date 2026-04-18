# neonka

```
$ md5sum data/train.raw data/train.lob.gz data/train.csv
8b9f8a864c77caa0e83523fa5a804b48  data/train.raw
cfadc1524cd19afda7a64b8f30405e32  data/train.lob.gz
0d92bf14fc90c1f89525d47bd31052e6  data/train.csv
```

## Dataset shape

| property | value |
|---|---|
| Total rows | 23,532,026 |
| **Sessions** | **63** |
| Avg session length | 373K ticks (min 146K, max 799K, CV 30%) |
| **Events per tick** | **0.899 ± 0.032 (CV 3.5%)** |
| Encoded price range (aR[0]) | 32446–40660 (these are `price × 4`, see convert.c) |
| **Real price range** | **8111 – 10165** (median 9388.5) |
| **Real tick size** | **0.5** in the original currency units |
| Median spread | 8 encoded half-ticks = 4 encoded ticks = 2.0 real tick units |
| Size aS[0] | median 2, max 245 |
| y label | **= mid[t+55] − mid[t]** (exact, slope 1.000, corr 1.00000) |

### Interpretation: 63 NYSE-style trading days

- **Sessions = trading days.** Boundaries between sessions show large price gaps
  (mean 0, std 127 half-ticks, up to ±650 half-ticks) — consistent with
  overnight/weekend moves between days.
- **Stable intraday pace.** Activity rate = **0.9 events per tick** is nearly
  constant across all 63 sessions (3.5% CV). Each session trades at the same
  pace per unit time, just for different total durations.
- **Tick ≈ 60 ms.** If the instrument averages ~15 events/second, one tick =
  0.9/15 ≈ 60 ms. A 373K-tick session ≈ 6.2 hours — consistent with the
  US equity market day (9:30 AM – 4:00 PM ET ≈ 6.5 hours, with some shorter
  days likely holiday-close).
- **63 sessions ≈ 3 months of NY trading.** Median price 18.8K ticks suggests
  a liquid mid-priced name (e.g., a large-cap NYSE/NASDAQ equity) or a
  normalized futures contract. 10% "no-event" frames reflect occasional
  quiet periods at this high-frequency sampling.
- **Each session is an independent calibration problem** — no carryover
  from prior day. Per-session rate tables are the natural modeling unit.



    Think of the book as a 2-state queue for orders:
    - State T = "at top" (best level)
    - State D = "deep" (levels 1–7)
    - 4 transitions:
      - tp: new order born at T (rate a)
      - tm: T→gone (cancel or execute from top, rate μ·n_T)
      - dp: new order born at D (rate b)
      - dm: D→gone (rate ν·n_D)
      - (plus T↔D shifts — when best moves, top rolls into deep or vice versa)

## Pipeline

```
data/train.raw (int32×49 cols)  ─┐
data/sessions.raw                 │
                                  ▼
  session -s S                 ← filter to session S
     │
     ▼
  pairs                        ← adjacent-frame pairs
     │
  ┌──┴──────────────────────┐
  ▼                         ▼
 rates -B sp0              tp / dp / tm / dm
 rates -B sp0_imb          (arrival distributions)
  │                         │
  ▼                         ▼
 *.rates files             tp.own / dp.own / per-(sp,imb) tables

  session -s S
    │
    ▼
  onestep -m <dir> -T 55 -S 100 -R seed [-W window]
    │
    ▼
 interleaved (seed, sim) pairs → avgsess.sh → R² vs y
```

## Current best R² at T=55

| config                             | mean R²  | weighted |
|------------------------------------|----------|----------|
| **Poisson Gillespie + -W 30**      | **1.81%**| 1.88%    |
| Poisson Gillespie (no -W)          | 1.79%    | 1.86%    |
| (XTX "simple baseline")            | 2.20%    | —        |
| Hawkes (-H)                        | 1.75%    | 1.81%    |
| Queue-coupled (-Q)                 | 1.70%    | 1.77%    |
| Bernoulli-per-tick                 | 1.66%    | 1.72%    |

Top sessions: ses 61 @ 5.4%, ses 57 @ 4.5%, ses 62 @ 4.5%.

## Bugs found and fixed

1. **Rate scaling** — `target = -log(nr)` assumed Poisson (2.3 events/tick);
   real data has 0.9 events/tick. Changed to `target = 1 - nr`. **+0.33 pp R²**.
2. **apply_tp crossed book** — inside-quote distance could push `aR[0] ≤ bR[0]`.
   Clamp to `opp ± 2` or skip.
3. **apply_dp duplicate R** — match-check gated on `F[k]==0`, causing a new real
   level to be inserted at the same price as an existing fake level.
   Fix: promote fake→real on match.
4. **Fake-pad dm picks** — `apply_dm` was picking fake levels; F-flag now skips.
5. **Bad-run flag** — onestep marks sim rows: `y=0` good, `y=1` book_dep,
   `y=2` sp_explode. `avgsess.sh` filters flag≠0 before averaging.

## Simplest linear signal: `y ≈ −1.46 · imb0`

A single-feature linear predictor captures **R² = 1.31%** — essentially identical to our 5-feature or 21-feature Ridge. All "imbalance-shaped" features carry the same signal:

| feature | R² with y (3M rows) |
|---|---|
| imb0 = (aN0-bN0)/(aN0+bN0) | **1.31%** |
| aN0 - bN0 | 1.20% |
| -sp · imb0 / 2 (microprice adj) | 1.16% |
| imb0 · sp | 1.16% |
| sign(imb0) | 1.15% |
| imb0 · aN0 | 1.01% |
| imb1 (level-1 imbalance) | 0.30% |
| sp alone | 0.01% |

Physical interpretation: thick queue side gets drained (cancels + market orders) faster than thin side. When bid is thick (imb<0), ask depletes → best ask climbs → mid rises → `y > 0`. Negative coefficient.

Our sim at T=55 achieves R² ≈ 1.81% (mean), beating the simple `-1.46 · imb0` by ~0.5 pp. The extra signal comes from book evolution that feature-level Ridge can't replicate without explicit state simulation. The 0.4 pp gap to XTX's 2.2% baseline likely requires features outside the 8-level LOB snapshot (time-of-day, session regime).

## Structural findings

### Detailed-balance identity `tp · dp = tm · dm`

Holds across AND within sessions, within ~2% error. Equivalent statements:

- `tp/tm = dm/dp`  (add-vs-remove odds at top = remove-vs-add odds deep)
- `tp/(tp+tm) + dp/(dp+dm) ≈ 1`  (tp_frac + dp_frac)
- **4 event rates collapse to 3 degrees of freedom per bucket**

Physical meaning: in equilibrium, the rate of orders entering the book at the
top must balance the rate leaving from the deep (and vice versa), with
microscopic reversibility. Comes from the same market-maker-driven
two-sided liquidity that makes y symmetric.

Implications for modeling:

- For each `(sp, imb_bin)` bucket, calibrate any 3 of `{tp, tm, dp, dm}` and
  derive the fourth. Reduces calibration noise in sparse tails.
- Enforcing the constraint as a manifold projection on noisy per-bucket rates
  is a cheap regularizer — helps when a session or (sp, imb) combination has
  few events.
- Violation of the constraint in calibration flags data issues or a
  miscategorized event (e.g., resize `r` being lumped in with one of tp/tm).
- **tp_frac drifts within session** (our diagnostic: ses 57 decile 0 = 0.86,
  decile 9 = 0.61). This single 1-D regime parameter captures most
  within-session dynamics and could be a rolling feature to track.

- **tm(n) is non-monotonic** in queue count — peak at n=3–5, decreases for n≥7
  (HLR queue-priority signature). Linear `μ_c·n + λ_m` fits well at small n.
- **Cross-event PC1 = 56% of variance.** One "activity axis" with tp/tm positive,
  dp/dm negative — anti-correlated top vs deep.
- **Real data is tick-quantized**: 89% of pairs have exactly 1 event,
  Poisson would give 37%. Structural fidelity ≠ R² fidelity (Bernoulli fix hurts
  R² despite matching distribution).
- **Rate clustering ACF ≈ 0.55 at lag 50** (real), ≈ 0 (Poisson sim).
  Hawkes doesn't fix this for R² — timing correction doesn't help direction
  prediction at 55-tick integration.

## Tools / scripts

```
onestep.c        Gillespie simulator (core)
session.c        session filter
pairs.c          adjacent-frame pair emitter
rates.c          event counting per (sp[, imb]) bucket
tp.c dp.c tm.c dm.c   arrival-distance distribution extractors
state.c          predicate filter on pair stream
stride.c         subsampling filter
avgsess.sh       per-session R² evaluation (parallel-safe)
sweep.sh         full 63-session sweep via xargs -P
compare.py       34 structural diagnostic sections
classify_pairs.py   pair complexity classifier (tick-quantization test)
feat_ridge.py    rich-feature Ridge (baseline 1.31% LOSO)
calib_hawkes.py  per-session Hawkes α, β from ACF
qgen.py qgen_all.py   synthetic LOB generators (oracle)
ratediff.py      rate-table comparison across sessions
poolgroups.py    PC1 session clustering for rate pooling
refs/            annotated bibliography + PDFs
```

## What didn't help R²

- Hawkes self-excitation (any α, β, calibrated or fixed)
- Queue-coupled rates (`-Q`)
- 6-bin vs 3-bin imbalance conditioning
- PC1 group pooling of dp/dm rates
- Reveal padding from history (`-L`)
- Size-based imbalance (corr 0.90 with count-based)
- Rich Ridge features (21-feature vs 5-feature: same)
- Small MLP / Transformer (overfits at this SNR)

## What would move R² further

- **Ensemble sim + Ridge** at prediction level (different errors; not tried yet)
- **Reference-price jumps** (HLR Model III)
- **Cross-session training with session embedding** (session-specific noise)
- **Information outside 8-level book** (time-of-day, exogenous flow signals)
