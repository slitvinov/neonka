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
| **Sessions** | **62** (after `split.c` min-length 150K guard; legacy 63-session split oversplit one day) |
| Avg session length | 380K rows (min 230K, max 799K, CV 30%) |
| **Events per row** | **0.91 ± 0.05** |
| Encoded price range (aR[0]) | 32446–40660 (`price × 4`, see convert.c) |
| **Real price range** | **8111 – 10165** |
| **Real tick size** | **0.5** |
| Median spread | 2.0 real tick units |
| Size aS[0] | median 2, max 245 |
| y label | **= mid[t+55] − mid[t]** (exact) |

### Interpretation: MBP-style event log

Each row is one **commit point** in an aggregated-by-price depth-of-book feed
(CME MBP-10, NYSE Pillar, Eurex MDI style). Columns per level:
`askRate, bidRate, askSize, bidSize, askNc, bidNc` × 8 levels + y.

- **Row = event, not wall-clock time.** Session length scales with daily message
  count (146K quiet → 799K busy). Tick duration ranges ~30–160 ms if sessions
  are 6.5h.
- **Tick-quantized**: 89% of pairs have exactly 1 elementary event (`|ΔN|`
  sum), 10% zero, 0.5% multi. Poisson-Gillespie would give 37/43/20. `classify_pairs.py`
  confirms; `tickgen.py` reproduces via one-categorical-draw-per-tick.
- **Session boundaries** are detected in `split.c` by price-overlap <5/16
  between frames. This oversplit once in the raw 63-session version — session
  44 was really one day with a fast intra-day move that cleared the 8-level
  grid (44+45 combined = 365K rows ≈ median day length, mid-gap at boundary
  only +6). Fixed by adding a **min-session-length 150K** guard; now 62
  sessions (min length 230K).

### Book model

2-state queue per side (at best = T, deep = D), 4 + 4 elementary events:

| label | name | symbol in HLR (2015) |
|---|---|---|
| tp_a / tp_b | top limit-order arrival | `L⁺_0` |
| tm_a / tm_b | top depletion (cancel + market order) | `L⁻_0` |
| dp_a / dp_b | deep limit-order arrival | `L⁺_k` |
| dm_a / dm_b | deep cancellation | `L⁻_k` |

In MBP data, cancel and market-order execution are indistinguishable on the
tm side — both reduce aN/bN. Calibrated models use `tm = μ·n + λ_m`.

## Pipeline

```
data/train.lob.gz ─► decode ─► data/train.raw
                 ─► split  ─► data/sessions.raw

session -s S ─► pairs ─► rates -B sp0_imb ─► awk (tables.sh)
                                              │
                                              ▼
                                       tables/tp.a.imb{0..5}.rates, ...
                                       tables/{tp,dp}.own

session -s S ─► events ─► hawkes -b β -i I ─► tables/hawkes.params

session -s S ─► onestep -m tables -T 55 -S 100 [-W 30] [-M params]
             ─► interleaved (seed, sim) pairs ─► python R²
```

Parallel wrapper: `seq 0 62 | sh para.sh sh <script>.sh {} [args with {}]`.

## Current best R² at T=55 (62 sessions, K=50 seeds averaged)

| method | mean R² | weighted | ≥0 sess | params/session |
|---|---|---|---|---|
| XTX baseline (reference) | 2.20 % | — | — | — |
| **8-D Hawkes MLE β-opt** (`-M`) | **+0.99 %** | **+1.05 %** | **48/62** | 73 (8 μ + 64 α + 1 β) |
| imb-bucket (`-W 30`) | +0.33 % | +0.42 % | 39/62 | ~3,000 (6 imb × SP × 8) |
| hybrid `-Y imb+α` | −4.20 % | −4.50 % | 2/62 | broken (see below) |

**8-D Hawkes wins 47/62 sessions over imb baseline** with 40× fewer parameters
(mean Δ = +0.66 pp). Fit via Ogata EM in C with golden-section β-search; all
62 fits in parallel take **7 s**.

The `-Y` hybrid (imb baseline + α·φ overlay) is broken because α was fit
assuming a constant-μ baseline absorbs everything; reusing it on top of imb
double-counts and over-drives the Gillespie clock. Fixing it requires refitting
α as a **residual-Hawkes** model given imb as exogenous baseline — separate
algorithm, not implemented.

## Simplest linear signal: `y ≈ −1.46 · imb0`

Single-feature R² = 1.31% — imb0 alone explains most of what Ridge/MLP can
extract without state simulation. All imbalance-shaped features collapse onto
the same direction:

| feature | R² (3M rows) |
|---|---|
| imb0 = (aN0−bN0)/(aN0+bN0) | **1.31%** |
| aN0 − bN0 | 1.20% |
| sign(imb0) | 1.15% |
| imb1 (level-1 imbalance) | 0.30% |
| sp alone | 0.01% |

Physical: thick queue drains fastest (cancels + market orders); bid-thick ⇒ ask
depletes ⇒ ask moves up ⇒ mid rises ⇒ y > 0.

## Structural findings

**Detailed-balance `tp · dp = tm · dm`** holds within 2% across and within
sessions. Equivalent: `tp_frac + dp_frac ≈ 1`. Reduces 4 rates to 3 degrees of
freedom per bucket — can derive the 4th rate analytically.

**tm(n) non-monotonic** in queue count — peak at n=3–5, decreases for n≥7
(HLR queue-priority signature).

**Cross-event PC1 = 56% of variance** — one "activity axis" with tp/tm
positive, dp/dm negative. This cross-coupling is what 8-D mutual-exciting
Hawkes captures (and what scalar self-excitation cannot).

**Rate clustering ACF ≈ 0.55 at lag 50** in activity rates (real); ≈ 0 in
Poisson sim. 8-D Hawkes gets partial recovery.

**Within-session drift**: `tp_frac` trends across deciles (ses 57: d0=0.86 →
d9=0.61). Current calibration is stationary per session — misses this.

## Tools / binaries

**C binaries:**
```
split       session-boundary detector
session     extract one session from train.raw
pairs       adjacent-frame pair emitter
rates -B X  per-bucket event counts (sp0 | sp0_imb | sp0_n0)
tp / dp     arrival-distance distributions
events      packed event log (for Hawkes MLE)
hawkes      8-D mutual-excitation Hawkes MLE (Ogata EM)
onestep     Gillespie simulator (-M 8-D Hawkes, -W local nr, -X qr, -H scalar)
replay / pack / state / stride / flip / center / offset
convert ↔ csv   bit-exact raw/CSV roundtrip
```

**Shell wrappers:**
```
bootstrap.sh   decode + split
tables.sh S D  build per-session calibration tables (6 per-sp + 54 per-imb + qr)
hawkes_fit.sh  fit 8-D Hawkes for one session
hawkes_sim.sh  run onestep with -M for one session × K seeds
imb_sim.sh     baseline imb-mode sim for one session × K seeds
simulate.sh    short-horizon onestep (T=1)
analyze.sh     compare.py report per session
classify.sh    event count distribution per session
para.sh        parallel wrapper (seq ... | sh para.sh sh <script>.sh {})
```

**Python:**
```
compare.py          23-section real-vs-sim diagnostic
classify_pairs.py   tick-quantization event classifier
feat_ridge.py       Ridge baseline (1.31% LOSO)
qgen.py / tickgen.py synthetic generators (Gillespie / tick-quantized)
calib_hawkes.py     ACF moment-match (superseded by hawkes.c MLE)
```

## What would move R² further

- **Residual-Hawkes refit** — fit α given imb as the exogenous baseline (so α
  only captures the residual temporal excitation, not absorbed by spatial
  bucketing). This is the correct version of the currently broken `-Y` hybrid.
- **Per-(sp, imb) μ_c with shared 8×8 α** — spatial μ, temporal α, joint MLE.
- **Ensemble** of 8-D Hawkes + imb predictions (different error structures).
- **Within-session regime features** — `tp_frac` drift hints at a 1-D rolling
  regime variable.
- **Information outside the 8-level book** (time-of-day, cross-session).
