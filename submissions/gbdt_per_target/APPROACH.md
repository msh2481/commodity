# GBDT per-target

## Core hypothesis

Nonlinear interactions and conditional patterns exist in the feature space that a linear model misses — certain features matter only in certain regimes, thresholds trigger asymmetric responses, etc. GBDTs are the industry workhorse for exactly this kind of noisy, heterogeneous, mid-sized tabular problem and should produce a strong tabular baseline. If a substantial feature set plus GBDT gives the same score as a trivial feature set plus GBDT, your bottleneck is feature engineering, not the model.

## Method

- **One GBDT per target** (424 models).
- **Primary library: XGBoost** with `device="cuda"`, `tree_method="hist"`. Tight RAPIDS integration, accepts `cudf.DataFrame` directly, well-documented GPU path. Alternative: LightGBM with `device_type="cuda"` — also valid but less natively RAPIDS-aligned.
- **Features.** GBDTs handle redundant / correlated features gracefully, so be generous. Start with:
  - Lagged log-returns of each underlying asset at 1, 5, 20, 60 days
  - Rolling volatility (std of returns) at 5, 20, 60 days
  - Rolling mean return at 5, 20, 60 days (momentum)
  - Cross-sectional rank per date of each asset's short-horizon return (relative momentum)
  - Rolling betas of each asset vs a market proxy (e.g. a broad-equity ETF's return)
  - For spread targets specifically: z-score of the spread itself (mean-reversion signal — literally what you're predicting on pair targets)
  - Day-of-week / month-end indicators (inferable from the contiguous date_id sequence)

## GPU libraries

- `xgboost` (>= 2.0, uses `device="cuda"` syntax; older `tree_method="gpu_hist"` also works)
- `cudf` for data loading and feature engineering
- `cupy` for numerical glue

Install:
```
pip install xgboost cudf-cu12 --extra-index-url=https://pypi.nvidia.com
```

## Progression

1. **Pipeline sanity.** XGBoost with default params on a tiny feature set (3-5 features), 100 trees. Confirm `./validate` runs on GPU end-to-end; verify via `nvidia-smi` that the GPUs are actually used during fold training.
2. **Full feature set.** Use the feature list above, with default-ish hparams (`max_depth=6`, `learning_rate=0.05`, `n_estimators=500`, `min_child_weight=5`). Record this as your within-thread baseline.
3. **Regularize for small data.** Each fold has ~1470 training rows per target. Test shallow trees (`max_depth=3-5`), heavier regularization (`gamma`, `reg_lambda`), early stopping on inner validation.
4. **Hparam search.** Manual first (depth × lr × n_estimators grid spanning extremes), then skopt with EIps once you have ≥ n_hparams points per the README conventions.

## Baselines to beat

- Zero predictor (pipeline sanity check)
- Mean-per-target predictor (best constant ranker)
- Your own step-2 GBDT (minimal-feature baseline) — each subsequent version should beat this

## Gotchas

- **424 models per fold is a lot.** Serial training on GPU might bottleneck. Consider training targets in parallel *within* the GPU (batched calls if possible), or accept 424× serial GPU calls and keep trees small to amortize warmup.
- **GPU warmup is real.** For a 100-tree model on 1470 rows, warmup can be longer than training itself. Profile once before committing to full validation.
- **`cudf` ↔ `numpy` conversions** are surprisingly expensive. Keep data on-GPU through training.
- **Overfitting risk.** With <1500 training rows per target, it's easy to memorize. Monitor an inner validation split for early stopping. Don't over-tune `n_estimators` without early stopping.
- **`min_child_weight` default is too low** for this data size. Start with 5-20 to prevent tiny leaves.
- **Subsample / colsample.** `subsample=0.8`, `colsample_bytree=0.5-0.8` often helps on noisy financial targets.

## Reporting

Log per-target runtime summary (even rough) — 424 models can take a while, and cost-per-trial matters for hparam search. Include total fold runtime and `eval_time_seconds` in `journal.jsonl`.
