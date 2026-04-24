You are one of several agents working in parallel on a commodity return forecasting problem. Each agent owns a single solution thread; you own this directory. **Do not read any other thread's files** (`submissions/<other>/`) **or `validation/log.csv`** — the whole point of running diverse threads in parallel is that they stay independent. Cross-contamination defeats the point.

## Read first

1. `../../README.md` — problem overview, data layout, evaluation metric, how the validator works, the `Model` interface, and repo-wide conventions.
2. `./APPROACH.md` — the specific hypothesis and method for this thread. Follow the "Progression" section in order unless you have a clear reason not to.

## Where your work goes

- Submission files: `./v1.py`, `./v2.py`, … in this directory. Each defines a `Model` class per the interface in the README.
- Attempt log: `./journal.jsonl`. One line per attempt, including failures (with the reason). Schema is in the main README under "Journal format".
- Shared intermediate artifacts (cached features, derived labels, etc.): this directory. Keep everything thread-local.

## How to validate

From this directory:

```bash
../../validate "short description" v3.py
```

The repo-root `validate` script is cwd-agnostic; you can call it from anywhere in the repo. It prints per-fold and overall Sharpe + `eval_time_seconds`. Copy the numbers into your `journal.jsonl` immediately after each run.

## Workflow

1. **Sanity first.** Start with the trivial version from APPROACH step 1. Get the pipeline green end-to-end — small model, minimal features, just make `./validate` run and produce a non-degenerate Sharpe — before doing anything ambitious.
2. **One hypothesis per version.** Each `vN.py` should isolate a single change. If you bundle feature changes and hparam changes in one diff, you can't attribute the delta.
3. **Log everything.** Every run gets a journal line. Failed runs too — note the failure mode.
4. **Hparam search.** Manual spread-across-axes first. Only switch to skopt once you have ≥ n_hparams journal entries; fewer and the surrogate is unreliable.
5. **Spot-check predictions.** After a non-trivial run, load `../../validation/predictions_cache/<slug>.csv` and eyeball a few rows. Catches silent bugs (all zeros, all identical, NaN-dominated outputs) that a Sharpe score can mask.

## Dependencies

The approach likely needs GPU libraries (cuML / cuDF / XGBoost with CUDA / etc.). Install what you need via `pip install --break-system-packages ...` — the environment already has `pandas`, `numpy`, `scipy`, `cargo`/`pueue`, and Python 3.12. 4× RTX 4090 and 256 CPUs are available.

## Mode

Execute autonomously. Make reasonable assumptions on routine decisions and document them in `journal.jsonl`. If you hit a genuine blocker (environment problem, missing data, contradictory instructions), flag it; otherwise keep iterating.

**Goal:** maximize the overall Sharpe reported by `./validate` on this thread's approach.
