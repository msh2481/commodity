# Factor model + residual prediction

## Core hypothesis

Most cross-sectional variance in asset returns is driven by a small number of latent factors — broad equity market direction, commodity beta, USD strength, risk-on/off sentiment, etc. If you can predict factor returns well (few outputs, high signal-to-noise) and add back idiosyncratic per-asset residuals, you cover most of the structure with fewer parameters than a direct per-asset model. The decomposition also forces the two prediction problems to be attacked with appropriate tools: factor returns often have macro/regime-level predictability; residuals are closer to short-term autocorrelation and pair-specific mean-reversion.

## Method

1. **Compute asset-level log-returns** from raw prices in `train.csv`. This gives a matrix `R` of shape `(n_dates, n_assets)`.
2. **Fit PCA on `R`** (training dates only, per fold). Get:
   - Factor time series `F = R · W` (shape `(n_dates, K)`)
   - Loadings `W` (shape `(n_assets, K)`)
   - Residuals `ε = R − F · Wᵀ` (shape `(n_dates, n_assets)`)
   Try `K ∈ {5, 10, 20}`.
3. **Train a factor model.** Predict the K factor returns at lags 1/2/3/4 using features — can be as simple as multi-output ridge on lagged factor returns + macro features. Because K is small, you can afford a richer model here.
4. **Train a residual model.** Predict per-asset residuals at the same lags. Residuals have smaller magnitude and are closer to pure noise for most assets, so expect a low R² here — but it can still move the Sharpe needle via rank structure.
5. **Recompose asset return predictions**: `R_hat_{t+ℓ} = F_hat_{t+ℓ} · Wᵀ + ε_hat_{t+ℓ}`.
6. **Compose target predictions.** Parse `target_pairs.csv`: for single-asset targets, use the predicted asset return directly; for spread targets `A − B` at lag ℓ, use `R_hat_{A, ℓ} − R_hat_{B, ℓ}`.

## Key design choices

- **Rolling vs static PCA.** Factor loadings drift over time; a 2018 PCA may not describe 2024 regimes. Options: recompute PCA per fold (static within the training window), rolling window (e.g. last 500 days) with re-fitting, or expanding window. Start with per-fold static — simplest and already respects the time-ordering.
- **What to predict for the residual?** The vanilla option is "next-period residual." An alternative is a zero-mean random walk assumption (residual prediction = 0), which is a strong baseline; if your residual model can't beat zero, it's adding noise and should be dropped.
- **Lag separately or jointly?** Jointly (one multi-output model with K × 4 outputs for factors, or n_assets × 4 for residuals) is cleaner and uses GPU well. Separately is fine if the factor structure differs across horizons.

## GPU libraries

- `cuml.decomposition.PCA` — sklearn-compatible API, GPU-native.
- `cuml.linear_model.Ridge` for both factor and residual prediction models (multi-output natively).
- `cudf` / `cupy` for data manipulation.
- If you want a non-linear factor model: XGBoost GPU.

## Progression

1. **Pipeline sanity.** K=5, ridge-predicted factors, zero residual. Confirm the composition logic works end-to-end.
2. **Factor structure check.** Before tuning prediction models, look at the explained-variance ratio from PCA. Top 5 factors should explain a solid majority of variance for this asset mix (equities, metals, FX). If they don't, the decomposition isn't capturing much and the approach probably won't pay off.
3. **Factor-only model.** Zero residual. This is the lower bound of the approach — how much does just predicting factors get you?
4. **Residual model.** Add per-asset residual prediction. Expect a small marginal improvement; if it's zero or negative, drop it.
5. **K sweep.** Try K = 5, 10, 20. More factors = more variance captured by factors, less by residuals — but also more parameters in the factor prediction model.
6. **Sanity vs direct.** As a within-thread check, also fit a direct per-target or multi-output ridge on the same features and compare. If factor decomposition beats direct prediction, the structural hypothesis is confirmed.

## Baselines to beat

- **Zero predictor** (sanity).
- **Factor-only** (residual prediction = 0) — lower bound within this approach.
- **Residual-only** (factor prediction = 0) — alternative lower bound; if this beats factor-only, your factor prediction is worse than predicting zero, which is a red flag.
- **Direct multi-output ridge on the same features** — compared within this thread as part of step 6 above.

## Gotchas

- **PCA on training data only.** Fitting PCA on the full timeline leaks future info into the training fold's factor loadings. Fit per fold using only that fold's training dates.
- **PCA on returns, not prices.** Prices are non-stationary and dominated by long-term drift; the first "factor" of a price PCA is just the overall price level. Use log-returns.
- **Missing prices** produce NaN returns. Impute (carry-forward, zero-fill) consistently before PCA — PCA can't handle NaNs.
- **Sign-flip ambiguity.** PCA factors are only defined up to sign; the sign might flip between folds. Doesn't matter for prediction (both the loading and the factor flip together), but watch out if you're trying to interpret specific factors.
- **Factor prediction is easier than residual prediction, but factor variance is larger, so it dominates.** Don't over-optimize residual R² at the cost of factor R² — the contribution to target variance is weighted by factor variance.
- **Residual orthogonality.** Residuals are orthogonal to factors by construction at train time. Ensure you don't reintroduce factor structure through the residual prediction model (e.g. by including factor features in the residual model — that's OK, but then your residuals aren't purely idiosyncratic anymore).

## Reporting

Per README conventions — `journal.jsonl` with K, factor/residual model choices, explained-variance ratios, and baseline comparisons.
