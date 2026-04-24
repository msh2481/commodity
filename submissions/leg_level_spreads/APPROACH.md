# Leg-level return prediction + spread construction

## Core hypothesis

~80% of the 424 targets are spread log-returns `log(A_{t+lag}/A_t) − log(B_{t+lag}/B_t)` between two underlying assets. Across all targets, only roughly 100 unique assets appear as legs. If you predict each asset's lagged log-return directly, you get:

- **~400 outputs instead of 424** (100 assets × 4 lags), with **many more training rows per effective output** (every target that uses an asset as a leg contributes evidence toward that asset's behavior).
- **Internal consistency by construction**: a target that's the difference of two other targets will automatically have a predicted return equal to the difference of its legs' predicted returns.
- **A cleaner prediction target**: a single asset's log-return has simpler time-series structure than a spread (which inherits noise from both legs and the subtraction).

## Method

1. **Parse `target_pairs.csv`** to enumerate the unique underlying assets. Entries look like `LME_PB_Close` (single) or `LME_PB_Close - US_Stock_VT_adj_close` (spread — note the spaces around `-`). Strip whitespace carefully.
2. **Build leg-level labels** from the raw prices in `train.csv`. `train_labels.csv` only contains the composed target labels (single-asset returns and spread differences) — it does *not* give you per-asset labels directly. But every leg name referenced in `target_pairs.csv` (e.g. `LME_PB_Close`, `US_Stock_VT_adj_close`) is itself a column in `train.csv`, so you can reconstruct leg-level returns yourself. For each unique asset column and each lag in {1, 2, 3, 4}:
   ```
   leg_return(asset, t, lag) = log(price[asset, t+lag] / price[asset, t])
   ```
   These derived `leg_return` values are the **training labels** for the leg model (for ridge or GBDT alike). Once the leg model is trained, its predictions are composed per step 5 below to produce the `target_0..target_423` values that the validator compares against `train_labels.csv`.
3. **Fit a leg model** that takes features at date `t` and outputs predicted log-returns for each (asset, lag) combination. This is naturally multi-output: either
   - **One multi-output fit** across all 400 (asset, lag) outputs (good for ridge — single linear-algebra call).
   - **One model per asset, multi-output across 4 lags** (trades computational cost for asset-specific features / hyperparams).
4. **Compose target predictions** using `target_pairs.csv`: for a spread target `A − B` at lag `ℓ`, predicted value = `leg_return(A, ℓ) − leg_return(B, ℓ)`. For single-asset targets, just the asset's return.

## Model choice inside the thread

This thread is about the compositional structure, not the model family. Either linear or GBDT is fine for the leg model:

- **Ridge leg model.** Multi-output closed-form ridge across all 400 outputs in one GPU call. Fastest to iterate.
- **GBDT leg model.** ~400 XGBoost fits (or batched) — heavier but potentially stronger if non-linearity matters.

Start with ridge — the compositional hypothesis is testable with a linear leg model, and you iterate faster. Swap in GBDT afterwards if ridge looks bottlenecked by linearity.

**Features** to use on the per-asset prediction:
- Lagged log-returns of each underlying asset at 1, 5, 20, 60 days
- Rolling volatility (std of returns) at 5, 20, 60 days
- Rolling mean return at same horizons (momentum)
- Cross-sectional rank per date of each asset's short-horizon return
- Rolling betas of each asset vs a market proxy
- Day-of-week / month-end indicators

Note that spread-specific features (like z-score of a specific pair's spread) don't apply naturally here, because the leg model predicts single-asset returns, not spreads. That's part of the simplification — if spread-specific signals matter a lot, leg-level composition will miss them.

## GPU libraries

- `cudf` for parsing target_pairs.csv + building leg returns from prices
- `cuml.linear_model.Ridge` for the ridge variant (multi-output)
- `xgboost` with `device="cuda"` for the GBDT variant

## Progression

1. **Derive leg returns correctly.** Compute `leg_return` for a handful of assets and cross-check against the corresponding `target_pairs.csv` entries — for a target `A − B` at lag ℓ, `leg_return(A, ℓ) − leg_return(B, ℓ)` should match the label column modulo the small number of NaN cases. **Do this before training anything** — if your leg-return derivation is wrong, the whole thread is wrong.
2. **Trivial leg model.** Predict leg returns as zero (i.e. model = no model). Confirm the composition logic works end-to-end and the resulting target predictions are zero/degenerate.
3. **Ridge leg model, minimal features.** A handful of lagged-return features only, multi-output across all (asset, lag) pairs. Record this baseline.
4. **Full feature set.** Add rolling moments, cross-sectional ranks, betas, calendar features.
5. **Sanity vs direct.** As a within-thread check, also fit a direct per-target ridge on the same features and compare. If leg-level composition beats direct prediction, the structural hypothesis is confirmed. If it doesn't, the hypothesis is wrong for this data.

## Baselines to beat

- **Zero leg-return predictor** (sanity).
- **Direct per-target ridge on the same features** — compared within this thread as part of step 5 above.
- **Your own step-3 baseline** — each subsequent feature addition should beat this.

## Gotchas

- **Label derivation mismatch.** If your derived `leg_return(A, ℓ) − leg_return(B, ℓ)` doesn't match the competition target label, you have a bug (often: wrong lag convention, off-by-one, or price column mismatch). Fix before any modeling.
- **Lag alignment.** Lag ℓ means "price at t+ℓ relative to t." Make sure your labels and predictions line up on the same `date_id`. The target at date_id `t` represents the return `t → t+ℓ`, predicted using features known at `t`.
- **Price gaps.** Exchange holidays create NaN prices at various dates. `log(NaN / x)` propagates. Decide a policy (carry-forward last price / drop that (date, asset) / interpolate) — and be consistent when computing both features and leg labels.
- **Asset set drift.** Some assets appear in only some targets' pairs. Don't assume every column in train.csv is a leg (not all are). Use the explicit set derived from `target_pairs.csv`.
- **Consistency only holds if you use one unified leg model.** If you fit separate models per (asset, lag) combo with different feature sets, you lose the "internally consistent" guarantee — the compositional identity only holds at the prediction layer, and noise in independent fits can break additivity.

## Reporting

Per README conventions — `journal.jsonl` in this folder. Record the leg model family (ridge/GBDT), feature set, whether you used multi-output vs per-asset, and the within-thread comparison against the direct-per-target baseline from step 5.
