"""Pick top-K diverse configs from tuning trials for ensembling.

Strategy: start with top-Sharpe, add next configs only if they're sufficiently
distant from already-picked ones in (normalized) hparam space. Writes
ensemble_configs.json for v40.py to consume.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

TRIALS = Path("/root/commodity/submissions/gbdt_per_target/_tune/trials.jsonl")
OUT = Path("/root/commodity/submissions/gbdt_per_target/_tune/ensemble_configs.json")

PARAMS = ["n_estimators", "max_depth", "learning_rate", "min_child_weight",
          "subsample", "colsample_bytree", "reg_lambda", "reg_alpha", "gamma"]
LOG_PARAMS = {"learning_rate", "reg_lambda", "reg_alpha", "gamma"}


def normalize(trials):
    """Scale each hparam to [0, 1] (log for log-uniform)."""
    arr = np.zeros((len(trials), len(PARAMS)))
    for j, k in enumerate(PARAMS):
        vals = np.array([t["params"][k] for t in trials], dtype=float)
        if k in LOG_PARAMS:
            vals = np.log(vals + 1e-12)
        lo, hi = vals.min(), vals.max()
        arr[:, j] = (vals - lo) / max(hi - lo, 1e-9)
    return arr


def diverse_topk(trials, k=4, min_dist=0.4):
    """Greedy pick: highest Sharpe first, then add if hparam-distance > min_dist."""
    trials = sorted(trials, key=lambda t: t["overall_sharpe"], reverse=True)
    norm = normalize(trials)
    picked = [0]
    for i in range(1, len(trials)):
        d = min(np.linalg.norm(norm[i] - norm[p]) for p in picked)
        if d >= min_dist:
            picked.append(i)
        if len(picked) == k:
            break
    return [trials[i] for i in picked]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--min-dist", type=float, default=0.35)
    ap.add_argument("--min-sharpe", type=float, default=0.20)
    args = ap.parse_args()

    trials = []
    for line in TRIALS.read_text().splitlines():
        if not line.strip(): continue
        d = json.loads(line)
        if d.get("status") == "done" and d.get("overall_sharpe", -1) >= args.min_sharpe:
            trials.append(d)

    if not trials:
        print(f"No trials with Sharpe >= {args.min_sharpe}."); sys.exit(1)

    print(f"Candidate trials (Sharpe >= {args.min_sharpe}): {len(trials)}")
    picked = diverse_topk(trials, k=args.k, min_dist=args.min_dist)

    print(f"\nPicked {len(picked)} diverse configs:")
    for i, t in enumerate(picked):
        p = t["params"]
        print(f"  {i+1}. trial {t['trial']:>2}: Sharpe={t['overall_sharpe']:+.4f}  "
              f"ne={p['n_estimators']} md={p['max_depth']} lr={p['learning_rate']:.3f} "
              f"mcw={p['min_child_weight']} ss={p['subsample']:.2f} cs={p['colsample_bytree']:.2f}")

    configs = [t["params"] for t in picked]
    OUT.write_text(json.dumps(configs, indent=2))
    print(f"\nWrote {len(configs)} configs to {OUT}")


if __name__ == "__main__":
    main()
