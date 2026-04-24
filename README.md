# Mitsui Commodity Prediction Challenge

Kaggle competition: predict 424 future commodity return / spread series from global market data. The competition closed 2026-01-17, so this is a pipeline-building / post-mortem exercise — the private leaderboard is locked. Our own validator (`./validate`) is the authoritative score.

## Competition at a glance

- **Task.** Predict 424 target columns, one row per `date_id`, for each trading day in the scored period. Each target is a log-return at horizon `lag ∈ {1, 2, 3, 4}` days, defined in `data/target_pairs.csv` as either a single asset's return or a spread return `A − B` between two assets. 106 targets per horizon × 4 horizons = 424.
- **Metric.** `mean(ρ_t) / std(ρ_t)` — Sharpe-style ratio where `ρ_t` is the Spearman rank correlation between predictions and realized targets across the 424 targets on day `t`. Consequences: (a) only cross-sectional *ordering* on each day matters — any monotonic transform of predictions leaves the score unchanged; (b) *consistency* matters more than peak accuracy — low-variance days with small positive ρ beat high-variance days with occasional large ρ.
- **Data span.** 1961 daily rows in train (`date_id` 0..1960).

## Data

All under `data/` (downloaded via `kaggle competitions download -c mitsui-commodity-prediction-challenge` and extracted).

| File | Shape | Notes |
|---|---|---|
| `train.csv` | 1961 × 558 | `date_id` + 557 feature cols |
| `train_labels.csv` | 1961 × 425 | `date_id` + `target_0`..`target_423` |
| `target_pairs.csv` | 424 rows | `target, lag, pair` — definition for each target |

### Features (557 cols in `train.csv`)

- **4 LME** closes — aluminum (AH), copper (CA), lead (PB), zinc (ZS)
- **40 JPX** — Gold Mini/Rolling-Spot/Standard + Platinum Mini/Standard + RSS3 Rubber futures, OHLCV
- **475 US** — ETFs and equities, `*_adj_close` and `*_adj_volume`
- **38 FX** — currency pairs

### Things worth knowing about the data

- **~10% of target cells are NaN** (a pair leg has no price that day due to exchange holidays / timezone differences). The scorer masks per-date NaNs before computing Spearman, so models can output anything for those cells — NaNs returned by a model are filled with 0 by the runner.
- **Targets are spread log-returns.** Most of the 424 are `log(A_{t+lag}/A_t) − log(B_{t+lag}/B_t)` — differential returns between two assets. This is structurally a stat-arb / pairs-trading problem.
- **`date_id` is a sequential integer, not a date.** 0..1960 in train. No calendar metadata is provided, but day-of-week / month-end effects can be inferred from the sequence if useful.
- **Missing exchange days.** Because of timezone / holiday differences, individual series have gaps within the shared `date_id` grid. Feature engineering that assumes contiguous history per series must handle this.

## Validator

### Usage

```bash
./validate "short description of this run" submissions/<thread>/<attempt>.py
```

This runs 4-fold cross-validation over the 1961 training dates, parallelized via `pueue`, and appends a row to `validation/log.csv`. The stdout summary includes per-fold and overall Sharpe, plus a `last-fold-delta` line (see *Validation scheme* below).

### Model interface

Your submission is a Python file defining a class `Model`:

```python
import pandas as pd

class Model:
    device = "cpu"           # "cpu" or "cuda"; default "cpu"
    cpus_per_fold = None     # optional int; default N_CPUS // 4

    def __init__(self) -> None:
        ...

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        # X: features for training dates, indexed by date_id (557 cols from train.csv)
        # y: labels  for training dates, indexed by date_id (424 target cols, may contain NaN)
        ...

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        # X: features for held-out dates, indexed by date_id
        # return: DataFrame indexed by date_id with columns target_0..target_423
        ...
```

A **fresh `Model()` is instantiated per fold**. Seeds (`random`, `numpy`, `torch` if imported) are set to 42 before `fit` runs. Own your own feature engineering — the runner passes raw columns from `train.csv` as `X`.

Sibling imports work: the directory containing your submission is added to `sys.path` before the file is loaded, so `from utils import ...` will resolve modules from the same subfolder.

See `submissions/dummy_random.py` for a minimal working example (Sharpe ≈ 0, as expected for independent random predictions).

### Compute scheduling

The validator uses [`pueue`](https://github.com/Nukesor/pueue) to queue fold jobs:

- **GPU models** (`device = "cuda"`): one GPU per fold, round-robin across pueue groups `gpu0..gpu3` (parallel=1 each). With 4 GPUs and K=4 folds, all folds train simultaneously. `CUDA_VISIBLE_DEVICES` is set per subprocess.
- **CPU models** (`device = "cpu"`): single `cpu` group, parallel=4. Each fold gets `N_CPUS // K` threads via `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`.

Multiple `./validate` invocations can run concurrently — pueue queues them against the shared groups, so launching several validations in parallel is safe. Check `pueue status` to see what's running / queued.

### Outputs

```
validation/
├── fold_assignments.csv        # fixed partition over date_ids; generated on first run
├── log.csv                     # append-only: one row per run, with all fold metrics
├── predictions_cache/
│   └── <slug>.csv              # concatenated OOF predictions (date_id × 424 targets)
├── models_cache/
│   └── <slug>.py               # snapshot of the submitted model.py (main file only)
└── runs/
    └── <slug>/                 # per-fold working files (safe to delete)
        └── fold_{0..3}.csv
```

`log.csv` columns: `timestamp, slug, description, model_src, model_cache, preds_cache, overall_{sharpe,mean_rho,std_rho,n}, eval_time_seconds, eval_wallclock_seconds`, and per fold `fold_{i}_{sharpe,mean_rho,std_rho,n,runtime_seconds}`.

`eval_time_seconds` is the **sum** of per-fold runtimes (actual compute, excluding queue wait — parsed from pueue `start`/`end` timestamps), useful for EIps-style cost accounting. `eval_wallclock_seconds` is the **max** (slowest fold), which matches real time-to-result since folds run in parallel.

### Validation scheme

K=4 equal chronological folds — dates 0..489 (fold 0), 490..979 (fold 1), 980..1469 (fold 2), 1470..1960 (fold 3). Each fold trains on the other three.

- **For model *selection*** (is A better than B?), use per-fold Sharpes directly. Time-generalization penalty is roughly shared across models, so the relative ordering under k-fold matches what you'd get from walk-forward.
- **For absolute-score calibration** (how does this map to held-out / future performance?), look at the `last-fold-delta` — Sharpe(fold 3) − mean(Sharpe of folds 0–2). Averaged across several diverse models, this estimates the "predict the most-recent period" penalty. The most-recent fold is the closest in-sample proxy to the real forecasting period.

See `validation/scoring.py` for the exact per-date Spearman and Sharpe implementation. NaN targets/predictions on a date are dropped pairwise before computing `ρ_t`.

## Repo layout

```
commodity/
├── README.md                           # this file
├── validate                            # bash entrypoint
├── validate.py                         # main driver
├── validation/
│   ├── fold_worker.py                  # per-fold subprocess (invoked by pueue)
│   └── scoring.py                      # Spearman / Sharpe helpers
├── submissions/                        # one subfolder per solution thread
│   └── dummy_random.py                 # smoke-test Model
├── data/                               # competition data (gitignored)
├── runpod.sh                           # environment bootstrap
└── mitsui-commodity-prediction-challenge.zip
```

## Agent workflow

Each **solution thread** — a coherent line of investigation (a model family, a feature-engineering idea, an ensemble approach) — lives in its own subfolder under `submissions/`:

```
submissions/
├── lgbm_per_target/
│   ├── v1_raw_features.py
│   ├── v2_rolling_returns.py
│   ├── v3_tuned.py
│   ├── utils.py                # shared helpers for this thread
│   └── journal.jsonl           # attempt log (see below)
├── ridge_spread_signals/
│   └── ...
```

Keep all work for a single thread inside its subfolder. Don't reach into other threads' folders; cross-pollinate by reading their `journal.jsonl` and copying ideas.

### Before each validation run

1. **Sanity first.** Get a trivial version working end-to-end before scaling up: minimal features, tiny model, maybe a subset of dates. Confirm the predictions are non-degenerate (not all zeros, not all identical) and the pipeline runs without error. If your first "real" attempt takes >10 minutes per fold, you'll burn GPU hours debugging the wrong things.
2. **Compare to baselines.** If the thread brief specifies baselines to beat, or earlier entries exist in your own `journal.jsonl`, know the target before you start. A run that "seems fine" in isolation may be worse than a trivial baseline.

Work off **your own** `journal.jsonl` — don't read other threads' journals or `validation/log.csv` to compare. The value of running several threads is that they're independently diverse; cross-contamination defeats the point.

### Journal (per-thread JSONL)

Every attempt gets a line in `submissions/<thread>/journal.jsonl`, written by the agent (not the validator). At minimum:

```json
{"timestamp": "2026-04-21T12:00:00Z", "slug": "20260421T120000Z_lgbm-v3", "attempt": "v3", "description": "500 trees, lr 0.05, rolling 20d features", "params": {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 63, "rolling_windows": [5, 20]}, "fold_sharpes": [0.12, 0.09, 0.14, 0.08], "overall_sharpe": 0.107, "eval_time_seconds": 340, "status": "done", "notes": "..."}
```

Rules:
- Append one line per *attempt*, including failures. For `"status": "failed"`, put the reason in `"notes"` — negative results save the next agent from retrying dead ends.
- `params` shape can differ between attempts. A downstream reader should take the union of keys across rows (missing → NaN) to get a rectangular table.
- Include `eval_time_seconds` (copy the value from the validator's stdout summary or `log.csv` — it's the sum of per-fold compute time, not wallclock, which is what EIps should optimize against).
- The `slug` should match what `./validate` assigned (visible in stdout and `log.csv`) so you can cross-reference to the scored artifacts.

### Hyperparameter search

- **First `≥ n_params` trials: manual, spread across dimensions.** Pick diverse combinations spanning the axes you suspect matter (high/low values across each hparam). The goal is a reasonable initial coverage of the space.
- **After that: `skopt` with `EIps`.** Fit a GP on (`params`, `overall_sharpe`, `eval_time_seconds`) by reading the journal, get a proposal, run it, append the result. Loop. `EIps` optimizes score-per-second — important here because eval times vary a lot (a 30s model and a 30min model should not be evaluated at the same rate).
- Do not jump to skopt before you have at least as many observations as hyperparameters — the GP surrogate is unreliable below that.

### Other habits

- **Spot-check predictions.** After a run, load `validation/predictions_cache/<slug>.csv` and look at a few rows. Catches bugs the aggregate score can hide (all zeros, all identical, NaN-dominated, clearly-biased outputs).
- **Read fold variance honestly.** A single great fold doesn't beat four mediocre ones — Sharpe's denominator punishes inconsistency. Wildly uneven fold Sharpes are a warning sign about regime sensitivity.
- **Leakage paranoia.** Anything computed using forward information is a bug. Rolling features, normalizations, target encodings — double-check that the computation at `date_id = d` uses only `date_id < d` (or `≤ d` if the feature is same-day observable).
- **Cache your intermediate features.** If a feature-engineering pass takes minutes to compute and is stable, save it (e.g. `submissions/<thread>/features_v1.parquet`) rather than recomputing per fold.

## Conventions

- **One submission per file** under `submissions/<thread>/`. Name it so a future reader can tell what changed (e.g. `v3_add_momentum.py`, not `final_final_v2.py`).
- **Don't bypass the validator** — its stdout summary and the row it writes to `log.csv` are the only trustworthy scores for a run. Don't compute ad-hoc Sharpes with your own splits and claim them.
- **Don't modify `fold_assignments.csv`.** All runs must score on the same partition or logs become incomparable.
- **Descriptive `description` arg.** It goes into the log; make it something a future you will actually recognize.
- **Seeds are fixed by the runner** (42 for `random`/`numpy`/`torch`). If you use additional RNGs, seed them yourself inside `fit`.
- **NaN predictions are accepted** (filled with 0 + warning). Prefer returning your best guess over NaN.
- **Set `device = "cuda"` for heavy models** so the scheduler pins one GPU per fold. Otherwise you'll thrash CPU. If your CPU model wants more than `N_CPUS // K` threads, set `cpus_per_fold`.
