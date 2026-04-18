# References

Papers relevant to this LOB-simulation/prediction project. Ordered by relevance to our current work.

## Core model (most used in discussion)

**HuangLehalleRosenbaum2015_QueueReactive.pdf**
Huang, Lehalle, Rosenbaum (2015). *Simulating and Analyzing Order Book Data: The Queue-Reactive Model.* JASA 110(509).
arXiv:1312.0563
- Non-parametric lookup of event intensities as functions of queue size.
- Our observed non-monotonic tm_rate(aN[0]) is exactly their cancel-rate signature.
- They don't model imb jointly; condition on 4-bucket opposite-queue discretization.
- Cited predictions (see our tests): cancel concave-then-flat, market-order exp decay in n_T, limit-insertion flat at L1 / decreasing at L2-3.

**ContKukanovStoikov2014_OFI.pdf**
Cont, Kukanov, Stoikov (2014). *The Price Impact of Order Book Events.* J. Financial Econometrics 12(1).
arXiv:1011.6402
- Order Flow Imbalance. Linear regression of price change on signed queue-changes at best.
- Reports contemporaneous R² ≈ 65% at 10-second horizon.
- Foundational for the `tm = μ_c·n + λ_m` split that worked for us.

## Hawkes / self-exciting extensions

**Bacry2015_HawkesReview.pdf**
Bacry, Mastromatteo, Muzy (2015). *Hawkes Processes in Finance.* Market Microstructure and Liquidity 1(1).
arXiv:1502.04592
- Canonical review. 8-dim mutually-exciting Hawkes on {T±/L±/C±/mid-up/-down}.
- Cures our rate-clustering ACF=0.55 at lag 50.

**MorariuPatrichiPakkanen2021_StateDepHawkes.pdf**
Morariu-Patrichi, Pakkanen (2021). *State-Dependent Hawkes Processes.* Quant. Finance 22(3).
arXiv:1809.08060
- Kernels that switch on LOB state (queue imbalance, spread).
- Likelihood improvement ≈ 20–40% over standard Hawkes.

**Wu2019_QueueReactiveHawkes.pdf**
Wu, Rambaldi, Muzy, Bacry (2019). *Queue-Reactive Hawkes.*
arXiv:1901.08938
- Hybrid: queue-reactive intensities + Hawkes residuals.

**BacryJaissonMuzy2016_HawkesKernel.pdf**
Bacry, Jaisson, Muzy (2016). *Estimation of Slowly Decreasing Hawkes Kernels.*
arXiv:1412.7096
- EM-like non-parametric kernel fitter; O(N·K·L) per pass.

## Deep learning

**Zhang2019_DeepLOB.pdf**
Zhang, Zohren, Roberts (2019). *DeepLOB: Deep Convolutional Neural Networks for Limit Order Books.* IEEE TSP 67(11).
arXiv:1808.03668
- CNN+Inception+LSTM on 10-level raw LOB. ~78% F1 on FI-2010 at k=10.

**Briola2024_DeepLOBForecasting.pdf**
Briola, Bartolucci, Aste (2024). *Deep Limit Order Book Forecasting.*
arXiv:2403.09267
- NASDAQ multi-stock. LOBFrame codebase.
- Recommends evaluating on complete-transaction probability, not F1.

**Berti2025_TLOB.pdf**
Berti et al. (2025). *TLOB: Transformer for Limit Order Book Forecasting.*
arXiv:2502.15757
- Dual-attention transformer; +3–8 pp F1 over DeepLOB on FI-2010.

## Order-flow / microstructure features

**Xu2023_CrossImpactOFI.pdf**
Xu, Cont, Cucuringu, Zhang (2023). *Cross-Impact of Order Flow Imbalance in Equity Markets.* Quant. Finance 23(10).
arXiv:2112.13213
- Multi-level OFI; PC1 explains 89% of OFI variance across levels 1–10.

## Recent extensions

**BodorCarlier2024_QROrderSizes.pdf**
Bodor, Carlier (2024). *A Novel Approach to Queue-Reactive Models: The Importance of Order Sizes.*
arXiv:2405.18594
- Extends QR with order-size distributions.

## Papers NOT downloaded (need SSRN/journal access)

- **Stoikov (2018). *The Micro-Price.*** Quant. Finance 18(12). SSRN:2970694.
  - Martingale-projection of imbalance × spread. Adds ~0.5–1.5 pp R² as a Ridge feature.
- **Kolm, Turiel, Westray (2023). *Deep Order Flow Imbalance.*** Math. Finance 33(4). SSRN:3900141.
  - LSTM on stationary OFI beats LSTM on raw LOB levels. GitHub: https://github.com/... (check SSRN link)
- **Cartea, Jaimungal, Ricci (2014). *Buy Low Sell High: a High Frequency Trading Perspective.*** SIAM FM 5(1).
