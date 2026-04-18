# The generator behind `data/train.events`

Reverse-engineered from empirical tests (see `evidence.py` figures + compare.py
forward vs time-reverse / rev+side-flip probes).

## One-line summary

**Bid/ask-symmetric mutually-exciting Hawkes process with a power-law kernel,
driving a full limit-order-book with queue-reactive dynamics. We see only the
top 8 levels.**

## Kernel form: POWER-LAW, not exponential (Bacry-Jaisson 2016)

Empirical event-rate ACF on ses45 (see `sim/figs/19_kernel_acf.png`):

| Model | Form | R² |
|---|---|---|
| Single exponential (fitted β) | `0.05·exp(−0.00039·τ)` | **0.62** |
| Power-law (fitted γ)          | `0.15·τ^(-0.274)`      | **0.97** |
| Two-exponential mixture       | fast β=0.14 + slow β=0.0005 | 0.75 |

**Our sim's assumed β=0.05 exponential is off by 10× or more at τ≥100 rows**
(see `sim/figs/21_kernel_residual.png`). Real LOB kernels have long-memory
power-law decay; the generator almost certainly uses such a kernel.

## Important caveat: HP is an observation artifact, not a generator event

What we call "HP" events (types 8, 9 in our event stream) are NOT independent
event types in the generator. They are our label for the moment when a deep
book level, previously beyond our 8-level observation window, becomes visible
because a cascade shifted all shallower levels up.

The generator presumably has a **full book with many levels**, all evolving
under the same Hawkes dynamics. We only observe the shallowest 8. Treating HP
as a generator event type is misspecification — it absorbs unseen deep-level
dynamics into fake parameters. We therefore fit the Hawkes on **types 0..7
only**. The sim's refill mechanism (`refill.c` + `onestep.c`) is a
bookkeeping hack to keep the truncated book well-defined, not a model of
real physical events.

## State

```
Book (per row):
  aR[0..7], bR[0..7]     — prices at each level
  aN[0..7], bN[0..7]     — order counts
  aS[0..7], bS[0..7]     — queue sizes

Hawkes memory:
  φ_c(t), c = 0..7        — exponentially-decayed event count per observed type
```

## Intensity

```
λ_c(t) = μ_c(state) + Σ_j α[c, j] · φ_j(t)
φ_j(t) = Σ_{past events of type j} exp(−β (t − t_event))
```

- `α[c, j]` is bid/ask-symmetric under `a ↔ b` swap (verified).
- `β ≈ 0.05` single global decay (half-life ≈ 14 rows).
- `μ_c` is state-dependent on (spread, imbalance, queue size); the
  queue-reactive part follows Huang–Lehalle–Rosenbaum (2015): 
  `rate_tm_a ≈ q_mu_a(sp, imb) · aN[0]`.

## Event types (8 visible)

| id | name | effect |
|---|---|---|
| 0 | tp_a | new ask, distance `d` inside spread (or queue add at `d=0`) |
| 1 | tp_b | same on bid |
| 2 | tm_a | remove one unit of top-ask queue (cascade if empties) |
| 3 | tm_b | same on bid |
| 4 | dp_a | insert new deep ask level |
| 5 | dp_b | same on bid |
| 6 | dm_a | remove a deep ask level |
| 7 | dm_b | same on bid |

(Types 8, 9 "HP" are observation artifacts — see caveat above; the fit and
sim both ignore them.)

## Simulation loop (Ogata thinning)

```
while t < T:
    λ* = Σ_c λ_c(t+)                      # upper bound at current time
    dt = Exp(λ*)
    t += dt
    φ *= exp(−β · dt)                     # decay memory
    λ_new = Σ_c λ_c(t)
    if uniform() · λ* < λ_new:            # accept
        pick event type ∝ λ_c(t)
        apply event to book
        φ[picked] += 1
```

This matches our `onestep.c` implementation (Ogata-thinned variant).

## Fitted parameters across 62 sessions

Fit: strict 8-D (HP-filtered input), β fixed at 0.05, EM with 500 iterations
and tolerance 1e-7.

| Parameter | Mean | Std | Range |
|---|---|---|---|
| β (kernel decay) | 0.05000 | 0.00000 | [0.0500, 0.0500] (hardcoded) |
| Σμ (baseline total rate) | 0.180 | 0.052 | (session-dependent) |
| ρ(α/β) (branching ratio) | **0.768** | 0.055 | [0.67, 0.93] |
| Σλ_stat (stationary rate) | 0.917 | 0.029 | matches observed 0.94 |

**Bid/ask symmetry of μ (ask/bid ratio):**

| pair | mean ratio | std |
|---|---|---|
| μ(tp_a)/μ(tp_b) | 1.016 | 0.134 |
| μ(tm_a)/μ(tm_b) | 1.009 | 0.125 |
| μ(dp_a)/μ(dp_b) | 1.152 | 0.383 |
| μ(dm_a)/μ(dm_b) | 1.129 | 0.314 |

**Key takeaways:**
- β is essentially a fixed constant (std/mean < 1%) — likely hard-coded.
- α structure is stable across sessions (std(ρ) ≈ 4%).
- Only μ varies per session, driving the vol regime split at ses52.
- tp/tm bid-ask symmetry <2% per session (strong generator symmetry).

## Empirical fingerprints (see `sim/figs/`)

| Test | Observation |
|---|---|
| Pooled P(Δ=+1)/P(Δ=−1) | 1.00083 (z=−0.32) → perfect symmetric generator |
| Cumulative bid vs ask event counts | tightly tracking; gap scales as √N |
| Time-reverse compare.py | marginals identical; price impact 2–4× smaller → **causal** |
| (Time-rev × bid-flip) compare.py | marginals and skew restored → combined symmetry |
| Section 29 price impact forward | +0.30 half-ticks at lag ≥ 5 → Hawkes self-excitation |
| Refill distance distribution | delta-dominated at 2 for most sessions, diffuse in ses52–61 |

## Literature

**Core Hawkes in LOB:**
- **Bacry E., Mastromatteo I., Muzy J.-F. (2015). *Hawkes Processes in Finance*.**
  Market Microstructure and Liquidity 1(1). [arXiv:1502.04592]
  Canonical 8-dim mutually-exciting Hawkes on {T±/L±/C±} events.

- **Morariu-Patrichi M., Pakkanen M. S. (2021). *State-Dependent Hawkes Processes
  and Their Application to Limit Order Book Modelling*.** Quantitative Finance
  22(3). [arXiv:1809.08060]
  Kernels that switch on LOB state (queue imbalance, spread). 20–40%
  likelihood gain over standard Hawkes.

- **Wu W., Rambaldi M., Muzy J.-F., Bacry E. (2019). *Queue-Reactive Hawkes
  Models for the Order Flow*.** [arXiv:1901.08938]
  Hybrid: queue-reactive baseline + Hawkes residuals.

- **Bacry E., Jaisson T., Muzy J.-F. (2016). *Estimation of Slowly Decreasing
  Hawkes Kernels: Application to High-Frequency Order Book Dynamics*.**
  [arXiv:1412.7096]
  EM-like non-parametric kernel fitter with slow decay (power-law tail).

**Queue-reactive dynamics:**
- **Huang W., Lehalle C.-A., Rosenbaum M. (2015). *Simulating and Analyzing
  Order Book Data: The Queue-Reactive Model*.** JASA 110(509).
  [arXiv:1312.0563]
  Non-parametric cancel/submit rates as functions of queue size. The `q_mu`,
  `q_nu` tables in our pipeline are their canonical parameterization.

- **Bodor B., Carlier G. (2024). *A Novel Approach to Queue-Reactive Models:
  The Importance of Order Sizes*.** [arXiv:2405.18594]
  Extends QR with order-size distributions.

**Order flow imbalance (for ML baselines, not used in generator):**
- **Cont R., Kukanov A., Stoikov S. (2014). *The Price Impact of Order Book
  Events*.** J. Financial Econometrics 12(1). [arXiv:1011.6402]
  Linear regression of price change on signed queue-changes at best.
  Foundational for the `tm = μ_c · n + λ_m` split.

- **Xu K., Cont R., Cucuringu M., Zhang J. (2023). *Cross-Impact of Order Flow
  Imbalance in Equity Markets*.** Quantitative Finance 23(10).
  [arXiv:2112.13213]
  Multi-level OFI; PC1 explains 89% of OFI variance across levels 1–10.

**Ogata thinning (simulation method):**
- **Ogata Y. (1981). *On Lewis' Simulation Method for Point Processes*.**
  IEEE Transactions on Information Theory IT-27(1), 23–31.
  The standard thinning algorithm for sampling a point process from its
  time-varying intensity function. Our `simulate()` in `onestep.c` is a
  direct implementation.
