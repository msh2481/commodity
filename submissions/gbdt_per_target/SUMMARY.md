# GBDT-per-target thread summary

**Best: v41, overall Sharpe 0.2376** (session 2: K-trick + BO-tuned ensemble).
Prior session 1 best was v32 = 0.2191. Session 2 added three independent gains: structural target masking, a slightly larger feature set, and a 384-trial skopt search over XGBoost hparams.

| Milestone | Sharpe | Key change |
|---|---:|---|
| v1 baseline | 0.1571 | 8 LME feats, default hparams |
| v32 (end of session 1) | 0.2191 | 4-way weighted ensemble |
| v37 | 0.2255 | v32 ensemble + K-trick mask (lag≥3) |
| v38 (session 2 baseline) | 0.2012 | reduced-features single model + K-trick (lag≥4 & has_LME) |
| v40 | 0.2351 | ensemble of 4 diverse BO-tuned md=3 configs |
| **v41** | **0.2376** | v40 + fold-3 hedge (trial 61, md=4) + region hedge (trial 14, md=8) |

v32 is a weighted 4-way ensemble: `v15(2) + v17(2) + v28(3) + v1(1)`. All four components are per-target XGBoost regressors (one model per of 424 targets), trained on `device="cuda"` with 100 trees, depth 4, lr 0.05, no subsample / colsample.

## Ranked results

Session 2 additions:

| Rank | Ver | Sharpe | Description |
|-----:|----:|-------:|-------------|
|  1 | v41 | +0.2376 | 5-way ensemble of tuned md=3 + fold-3 hedge + md=8 region hedge |
|  2 | v40 | +0.2351 | 4-way ensemble of diverse md=3 BO-tuned picks |
|  3 | v37 | +0.2255 | v32 ensemble + K-trick mask lag≥3 (212 targets) |
|  4 | v38 | +0.2012 | single-model BO-warmup baseline: 31 feats + default hparams + K-trick lag≥4 & has_LME |

Session 1 ranking (full sweep on all 424 targets):

| Rank | Ver | Sharpe | Description |
|-----:|----:|-------:|-------------|
|  1 | v32 | +0.2191 | 4-way weighted ensemble v15(2)+v17(2)+v28(3)+v1(1) |
|  2 | v31 | +0.2186 | 3-way weighted v15(1)+v17(2)+v28(3) |
|  3 | v30 | +0.2181 | 3-way equal v15+v17+v28 |
|  4 | v29 | +0.2162 | 3-way weighted v14(1)+v15(2)+v28(2) |
|  5 | v26 | +0.2110 | 3-way weighted v14(1)+v15(2)+v17(1) |
|  6 | v22 | +0.2103 | 2-way equal v14+v15 |
|  7 | v25 | +0.2083 | 3-way equal v14+v15+v17 |
|  8 | v28 | +0.2033 | v15 + minimal JPX common (Gold+Platinum Std) |
|  9 | v15 | +0.2019 | v14 + cal + multi-window (5/20/60) spread |
| 10 | v18 | +0.1985 | v15 + horizon-matched spread |
| 11 | v19 | +0.1983 | v15 bagged (3 seeds) |
| 12 | v21 | +0.1962 | v15 + subsample/colsample 0.8 |
| 13 | v14 | +0.1917 | v1 LME + per-target 20d spread |
| 14 | v17 | +0.1911 | v14 with spread window 5d |
| 15 | v23 | +0.1894 | v15 + Huber loss |
| 16 | v24 | +0.1891 | v15 windows 3/10/40 |
| 17 | v16 | +0.1880 | v15 features + deeper hparams |
| 18 | v27 | +0.1883 | v15 + LME vol20 in common |
| 19 | v20 | +0.1849 | v15 + all 8 JPX closes in common |
| 20 | v13 | +0.1782 | v1 LME + per-tgt A/B asset + spread |
| 21 |  v1 | +0.1571 | **baseline**: 4 LME close + 1d log-ret (8 features) |
| 22 | v11 | +0.1286 | v1 + LME 5d/20d returns (16 feats) |
| 23 | v12 | +0.1252 | v1-hparams + 861 broad feats + spread |
| 24 |  v9 | +0.1099 | per-target focused (A, B, pair, market proxies) |
| 25 |  v8 | +0.1007 | v1-hparams + 861 broad feats |
| 26 |  v6 | +0.0959 | v2 + per-target spread features |
| 27 | v10 | +0.0912 | v9 + raw log-price + more horizons |
| 28 |  v3 | +0.0766 | 861 feats, heavy reg |
| 29 |  v2 | +0.0700 | 861 feats, default-ish hparams |
| 30 |  v5 | +0.0700 | v2 with threading (determinism check) |
| 31 |  v7 | +0.0694 | 861 feats + heavy reg (shallow) |
| 32 |  v4 | +0.0560 | v2 + per-target pred z-score |

## Progression of key insights

1. **v1 baseline is shockingly strong.** Just 4 LME raw closes + their 1-day log-returns. Only 8 features. Scored 0.157 because the 4 LME tickers appear in ~295 of 848 pair legs in `target_pairs.csv` — so LME-centric features are relevant to a majority of targets.

2. **Broad feature dumps overfit.** v2 with 861 generic features (log-rets 1/5/20 + vol20 + mean20 + xs-rank for all 143 price-like cols) scored 0.070 with default hparams and 0.077 with heavy regularization. Fold 0 even went *negative* for most v2-family attempts. Conclusion: ~1470 rows per target cannot support 861-dim inputs, and the noise features hurt more than the marginally-relevant ones help.

3. **Prediction post-hoc z-scoring hurts (v4).** Normalizing each target's predictions by its own train-prediction std promoted low-signal targets and damped high-signal ones, scoring 0.056 — worst of all.

4. **Per-target ultra-focused feature sets (v9, v10) are too sparse.** Giving each target only its own legs' features + a few market proxies hit 0.11. Per-target focus loses cross-asset context the model was implicitly using.

5. **Hybrid wins (v13, v14).** Keep v1's 8 LME features as a shared common base, then *add* per-target pair-specific spread features (log(A/B), rolling dev, rolling z-score). v14 with just one 20d window scored 0.192. v13 additionally adding A/B asset-level features got 0.178 — adding too much per-target information diluted the good signal.

6. **Multi-window spread is the critical feature (v15).** Using windows (5, 20, 60) together plus cal features jumped single-model performance to **0.2019**. This captures short-, medium-, and long-term mean-reversion dynamics simultaneously.

7. **Alternative hparams, losses, and stochastic training are neutral or worse.** v16 (deeper, lower lr, heavier reg), v21 (subsample 0.8, colsample_bytree 0.8), v23 (pseudohuber loss), v19 (bagging 3 seeds) all scored *below* v15. Default-ish hparams on the minimal feature set is the sweet spot.

8. **Adding minimal JPX features helps in ensemble even when it barely helps alone (v28).** v28 = v15 + 6 JPX features (Gold_Standard + Platinum_Standard raw + ret1 + vol20). Scored 0.2033 alone — only marginal over v15. *But* added meaningful diversity to ensembles: v28 + v15 + v17 reached 0.218. Earlier v20 tried all 8 JPX columns in the common tail and scored 0.185 — more JPX is worse; just the two most-referenced JPX futures work.

9. **Ensemble wins plateau around 0.219.** Averaging 2-4 models trained on slightly different feature configs gains +0.015-0.020 over the best single. 5-model ensembles don't beat 4-model — more diversity just dilutes the best components.

10. **Fold 3 is structurally hard.** Across all variants, the last chronological fold (dates 1470-1960) scored 0.10-0.14, noticeably below folds 0-2 (0.15-0.27). Adding v1 to the ensemble with a small weight slightly improves fold 3 robustness at no cost elsewhere (v32 fold 3 = 0.137 vs v30 fold 3 = 0.130).

## Feature-engineering takeaways

- Raw prices matter. v1's raw LME close prices carried useful level/regime information the model used. Models with only log-returns (stationary features) consistently underperformed until spread level/z features added non-stationary info back.
- "Spread features" are the MVP addition. For each target `A - B`, computing `log(A_t / B_t)` and its rolling z-score gives the model the exact stat-arb signal most targets are built around.
- Calendar features (`date_id % 5`, `date_id % 22`, `date_id / 2000`) give a small but reliable lift when combined with spread features (v15 > v14 by ≈0.010).
- Multi-window is better than single-window. Windows (5, 20, 60) cover short/medium/long mean-reversion time scales. Alternative choices like (3, 10, 40) scored 0.189.
- Lag-matched spread features (v18, using horizon h as the lookback) did not beat multi-window.
- Per-asset volatility (LME vol20) in common tail hurt — too-noisy.
- XGBoost on GPU with ThreadPoolExecutor (max_workers=8) + per-fit `n_jobs=1` overlaps CUDA streams and gives ≈2-3× speedup over serial fits.

## Hparam takeaways

- `n_estimators=100, max_depth=4, learning_rate=0.05, min_child_weight=5, no subsample, no colsample` consistently beat every alternative I tried, including deeper trees (v16), heavier regularization (v3, v7), stochastic tree sampling (v21), and alternative losses (v23).
- Bagging with 3 different random seeds (v19) did not improve over single-seed, suggesting XGBoost is already near-deterministic enough on this data size.
- Skopt-based search was never triggered — my manual grid already covered enough of the hparam space that the single-model plateau was clearly at ≈0.20, and further gains came from feature/ensemble design, not hparam tuning.

## Files

Session 1 milestones:
- `v1.py` – minimal LME-only baseline (Sharpe 0.157).
- `v15.py` – best session-1 single model (Sharpe 0.202).
- `v32.py` – session-1 best, 4-way weighted ensemble (0.2191).
- `v37.py` – v32 + K-trick lag≥3 mask (0.2255).

Session 2 additions:
- `v38.py` – first reduced-target single model (0.2003), full feature pool incl. LME extensions.
- `v38_tune.py` – same pipeline as v38 but LME extensions dropped, loads hparams from `_tune/hparams.json`. Fed by the BO loop.
- `v40.py` – parameterized ensemble; reads a list of configs from `_tune/ensemble_configs.json`. v40 itself = 4 md=3 picks → 0.2351.
- `v41.py` – same code as v40, different configs file state (5-way with fold-3 hedge) → **0.2376**.
- `tune.py` – skopt orchestrator (GP + EIps, warmup-aware, batch-tell).
- `_tune/` – BO work dir: `hparams.json`, `ensemble_configs.json`, `trials.jsonl` (384 trials), `tune.log`, `plots/`, `analyze.py`, `pick_ensemble.py`.

Shared:
- `utils.py` – shared feature helpers.
- `ensemble_test.py` – post-hoc weight exploration on cached predictions.
- `journal.jsonl` – full attempt log (v1..v41 + tune trials).

## Session 2: K-trick target masking + BO-tuned ensemble (v37 → v41)

Three independent changes, each adding ~0.01 Sharpe.

### 1. K-trick — predict only a subset of targets, fill the rest

The Sharpe-of-Spearman metric is cross-sectional per date: targets where predictions have no signal act as pure noise in the ranking. Solution: predict only the targets where our features have structural signal, fill the rest with the per-day median of the predicted subset.

- **v37 mask (lag ≥ 3, 212 of 424 targets)** lifted v32 ensemble 0.2191 → 0.2255 (+0.006). Rationale: longer-horizon log-returns accumulate more signal relative to daily noise.
- **v38 onward mask (lag ≥ 4 AND has_LME, 71 targets)** — tighter, targeting only the regime where LME-heavy feature set dominates. Faster iteration (~1/3 runtime per fold) and each per-target model focuses on data it can actually learn. Single-model v38 scored 0.2012 with default hparams — close to v37's 0.2255 despite being un-tuned single model, because it only predicts meaningfully on the 71 targets.

Implementation: build a full D×T prediction array, fit per-target models only for kept indices, then broadcast the per-day median of kept preds into the filler columns. Structurally identical output shape.

### 2. Feature set refinement

Combined the "solid" features from session 1 into a common per-model feature pool (~31 feats for spread targets):

- Common tail (17): 4 LME raw + 4 LME ret1 + 2 JPX_Gold_Std + 2 JPX_Platinum_Std raw/ret1/vol20 + 3 calendar
- Per-target spread (10): current spread, dev & z at windows {5, 20, 60}, horizon-matched h-day & 2h-day returns, spread_vol20
- Per-target leg minimal (4): ret5 + vol20 for A and B

**Notable dropped:** LME extended horizons (ret5/ret20/vol20 — 12 feats). Ran an ablation: full-feature v38 = 0.2003, reduced v38 = 0.2012. Confirms LME extensions add only noise here, consistent with session 1's v11/v27 findings.

### 3. Bayesian optimisation of XGBoost hparams

384-trial skopt search with GP + EIps acquisition over 9 dims:

| Dim | Range | Prior |
|---|---|---|
| n_estimators | [50, 500] | uniform |
| max_depth | [3, 8] | uniform |
| learning_rate | [0.01, 0.2] | log-uniform |
| min_child_weight | [1, 50] | log-uniform |
| subsample | [0.5, 1.0] | uniform |
| colsample_bytree | [0.3, 1.0] | uniform |
| reg_lambda | [0.01, 50] | log-uniform |
| reg_alpha | [0.001, 10] | log-uniform |
| gamma | [1e-3, 1.0] | log-uniform |

**Gotchas found the hard way:**
- `gamma` and `reg_alpha` default upper bounds of [0, 5] and [0, 10] were far too large for this loss scale (target variance ≈ 1e-4); LHS samples with γ ≥ 0.1 effectively block every split. Fixed by switching both to log-uniform with small lower bounds.
- LHS seeding wastes 10+ trials in heavily-regularized corners. Injecting the v38 default-hparams baseline as a warmup point gave the GP a strong anchor. After that, LHS for remaining seed points.
- `opt.tell()` serialized in a loop refits the GP N times (74 iters ≈ minutes); batch-tell one list at once.

**Best single config: trial 350, Sharpe 0.2348** — `ne=446, md=3, lr=0.173, mcw=22, ss=0.89, cs=0.46, rl=0.04, ra=1e-3, g=1e-3`.

**GP partial-dependence findings:**
- `max_depth = 3` strongly preferred — reverses session 1's md=4 consensus. The short 12-LHS run that ended at 0.2265 had found a local optimum at md=8 / ne=50; with more trials the GP discovered a much flatter + broader optimum at md=3 / ne=400-500.
- `n_estimators` monotonically better toward the upper bound (500) — could widen the cap further.
- `gamma` and `reg_alpha` both want the floor value (~1e-3); effectively turned off.
- `learning_rate, min_child_weight, subsample, colsample_bytree, reg_lambda` all have weak partial dependence — largely fungible.

Plots saved in `_tune/plots/{convergence,evaluations,objective}.png`.

### 4. Ensembling the tuned configs

v40 (equal-weight k=4, all md=3 trials from top of the BO): 0.2351. Marginal lift (+0.0003 over best single). The top-4 are nearly-identical-region md=3 configs, so their errors are highly correlated — especially on fold 3 (last-fold-delta = −0.106).

v41 (k=5 mixed-depth): 0.2376. Added two hedges to v40:
- **Trial 61** (md=4, ne=54, lr=0.15): overall 0.188 alone but **fold-3 Sharpe 0.274** — best fold-3 performer with non-trivial overall.
- **Trial 14** (md=8, ne=50, lr=0.04): the session-1 winner (0.2265) from a totally different hparam region.

Fold 3 lifted 0.158 → 0.171; last-fold-delta improved to −0.091. The anti-correlation hedge paid off exactly as hypothesized.

## If continuing

- `n_estimators` upper bound is pinned at 500 in the BO; widen to 1000+ and re-run. Also consider raising `max_depth` upper to 12 now that md=3 is the known optimum — keeps the GP honest about its belief.
- Stacking with a meta-learner over the 4 ensemble components might break the 0.22 ceiling, but requires another fold-split for honest meta-training.
- Fold 3 weakness suggests a regime-shift in the most-recent data. Sample-weighting (recent dates heavier) could bias the trained model toward the forecast regime, but would trade off generalization to the other folds; under K-fold this risks inflating Sharpe on fold 3 only.
- A separate model class trained on horizon-grouped data (one model per lag ∈ {1,2,3,4}, with lag as a feature, pooling 4× rows per pair) could share information across lags. Not tried here.
- Non-GBDT diversity (ridge, nearest-neighbor regressor on asset embeddings) would add truly independent errors to the ensemble; this thread stayed GBDT-only per its brief.
