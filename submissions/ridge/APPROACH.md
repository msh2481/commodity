# Ridge (multi-target)

## Core hypothesis

A single multi-output ridge regression on well-engineered features captures most of the available signal in short-horizon returns. Linear-on-good-features is typically within ~15% of GBDT on noisy financial targets and lets you iterate on features an order of magnitude faster.

## Why multi-output, not per-target

For ridge, the closed-form solution decomposes exactly across outputs: `β = (XᵀX + λI)⁻¹ Xᵀ Y`. One matrix inversion solves all 424 targets simultaneously. Fitting 424 separate ridges would do the same work 424× with no benefit — and with tiny per-call GPU ops where warmup dominates.

The only reason to fall back to per-target fits would be differing NaN masks across targets (which we do have — label NaN rates vary by target). Handle this with one of:
- **Zero-impute NaN labels and do a single multi-output fit.** Mathematically this biases per-target predictions toward zero on rows that had NaN, but because the metric is rank-based and monotonic-invariant, a constant shift doesn't hurt. Simplest, fastest.
- **Sample-weight imputation: set weight=0 on NaN rows.** sklearn ridge only accepts 1D `sample_weight`, so this only works if you group targets by shared NaN pattern. Practical if there are a handful of dominant patterns.
- **Per-target fallback for the high-NaN tail.** Fit multi-output on the bulk with zero-imputation, then refit the worst ~20 targets individually with their NaN rows dropped.

Start with the zero-impute version — it's one line of code and probably good enough.

## Features

Same feature set across all 424 targets (that's the point of multi-output). Invest here:
- Lagged log-returns of each underlying asset at 1, 5, 20, 60 days
- Rolling volatility and rolling mean return at 5/20/60 days
- Cross-sectional rank per date (relative momentum)
- Rolling betas of each asset vs a market proxy
- For spread targets: z-score of the spread itself (mean-reversion signal)
- Day-of-week / month-end indicators (inferable from the contiguous date_id sequence)

Standardize features per column (or better: per date, cross-sectionally) before fitting. Doesn't change Spearman but keeps ridge well-conditioned.

## GPU libraries

- `cuml.linear_model.Ridge` — drop-in sklearn API, accepts `cudf.DataFrame` and `cupy.ndarray`.
- `cudf` for data loading and rolling features (much faster than pandas at this size for multi-horizon rolling ops).
- `cupy` for numerical glue.

Install:
```
pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12 cuml-cu12
```

If the dataset (1961 × 557 features × 424 outputs) ends up so small that GPU warmup dominates, fall back to `sklearn.linear_model.Ridge` on CPU. Profile once before committing.

## Progression

1. **Pipeline sanity.** Multi-output ridge with a tiny feature set (5-10 lagged returns), zero-imputed labels, default alpha. Confirm `./validate` runs and Sharpe is non-degenerate.
2. **Baseline bar.** Full classical feature set (multi-horizon returns + rolling vol + cross-sectional rank). Record this number.
3. **Spread-specific features.** Add z-scores of pair spreads for the spread targets. Note that you can't easily get per-target-category features into a shared multi-output fit without expanding to a "tall" format — if that constraint ends up limiting signal, consider whether the linear framing is the right one for this problem.
4. **Alpha sweep.** Single global alpha tuned on OOF CV performance.
5. **NaN refinement.** Only if results look limited: try per-target fallback for high-NaN targets.

## Baselines to beat

- Zero predictor (pipeline sanity)
- Mean-per-target predictor (best constant ranker)
- Step-2 ridge (your own thread baseline)

## Gotchas

- **Rolling-window leakage.** Features that use future info across fold boundaries will inflate CV scores. Compute rolling features on a per-fold basis using only the training dates' history, or use strictly causal (trailing-only) windows on the full timeline.
- **Feature NaNs from rolling ops.** Rolling windows produce NaN at the start. Trim or impute; be consistent across folds.
- **GPU ↔ CPU conversions** are expensive relative to the fit itself at this size. Keep features on-GPU through training.
- **Zero-imputation caveat.** Zero-imputing label NaNs for missing targets effectively tells the model "predict 0 when data is missing." That's fine for rank purposes but inflates MSE. Don't interpret the training MSE as meaningful.

## Reporting

Per README conventions — `journal.jsonl` in this folder, one line per attempt, including feature-set name, alpha, NaN strategy, and runtime.
