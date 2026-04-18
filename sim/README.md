# Synthetic-data evidence

Scripts & figures accumulating evidence that `data/train.events` is from a
symmetric synthetic generator, not a real market tape.

## Run

```sh
sh sim/run.sh
# → writes data/ .dat files and figs/ PNGs
```

## Headlines

| Test | Expected if real | Observed |
|---|---|---|
| Pooled P(Δ=+1) / P(Δ=−1) | tiny drift, z ∼ ±10 | **1.00083, z = −0.32** |
| Mean of per-session mean(Δmid) | non-zero | **+5×10⁻⁵ ticks/row** |
| Mean of per-session skew(Δmid) | non-zero | **+0.006** (std 0.53) |
| Bid↔ask cumulative-count drift | persistent (macro flows) | **bounded, √N-scale** |
| End-of-session tp_a−tp_b (mean over 62 sessions) | non-zero | **+121** (std 2740) |

## Figures (sim/figs/)

| File | Shows |
|---|---|
| 01_dmid_pooled.png | Pooled Δmid distribution (log y). Symmetric about 0. |
| 02_dmid_mirror.png | P(+k) vs P(−k) overlaid — indistinguishable in log scale. |
| 03_dmid_per_session_ratio.png | log₂(P(+1)/P(−1)) per session — zero-centered. |
| 04_sym_pooled_ratio.png | Pooled log-ratio at each |k| with z-scores. |
| 05_event_balance.png | tp/tm/dp/dm bid vs ask event counts per session — all ~0. |
| 06_fingerprints.png | Session vol / spread / event-rate — regime break at ses52. |
| 07_mid_return.png | Per-session mid drift — aggregate ~0. |
| 08_smoking_gun_cumcount.png | **Cumulative tp_a/tp_b/tm_a/tm_b curves track tightly.** |
| 09_smoking_gun_diff.png | Bid↔ask cumulative DIFFERENCE — bounded, mean-reverting. |
| 10_cum_gap_summary.png | max and end-of-session a−b gap across all 62 sessions. |
| 11–17 (params)         | Fitted Hawkes params per session (β, μ, ρ, λ_stat, α heatmap). |
| 18_refill_tail_fits.png | Pooled refill tail fits (power-law vs exp vs log-normal). |
| 19_kernel_acf.png      | **Kernel-form test:** event-rate ACF log-linear + log-log. |
| 20_kernel_per_session.png | Kernel form across 4 representative sessions (power-law wins). |
| 21_kernel_residual.png | Residual: empirical ACF / (our sim's β=0.05 kernel) — **exponential off by 10×+ at τ=100**. |

## Verdict

Strong evidence for **symmetric synthetic generator**, with:
- Per-event bid/ask coin-flip symmetry (pooled up/down +1 count matches to 0.03 %, z = −0.32).
- Session-level asymmetries ~3× the pure-Bernoulli expectation, but averaging to zero
  across the 62-session ensemble — consistent with each session being an iid draw from
  a symmetric process.
- Regime break at ses52 (vol/spread jump) looks engineered, not a natural market event
  (ses52 kurtosis = +1292).
