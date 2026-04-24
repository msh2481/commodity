"""Diagnostic: compute Sharpe of a weighted ensemble from predictions_cache.

Not for final scoring (use ./validate with a real ensemble Model for that).
Just for exploring which combinations are promising.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("/root/commodity/validation")))
from scoring import per_date_spearman, sharpe_from_rhos  # noqa: E402


PREDS_DIR = Path("/root/commodity/validation/predictions_cache")


def load_preds(slug: str) -> pd.DataFrame:
    path = PREDS_DIR / f"{slug}.csv"
    return pd.read_csv(path).set_index("date_id")


def rank_normalize(preds: pd.DataFrame) -> pd.DataFrame:
    """Per-target percentile rank across the cached date range."""
    return preds.rank(pct=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slugs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float)
    ap.add_argument("--rank-normalize", action="store_true",
                    help="z-score predictions per target (rank pct) before averaging")
    args = ap.parse_args()

    pred_list = [load_preds(s) for s in args.slugs]
    weights = args.weights if args.weights else [1.0] * len(pred_list)
    if len(weights) != len(pred_list):
        raise SystemExit("weights length must match slugs")

    if args.rank_normalize:
        pred_list = [rank_normalize(p) for p in pred_list]

    # Align indices
    common = pred_list[0].index
    for p in pred_list[1:]:
        common = common.intersection(p.index)

    stacked = np.zeros((len(common), pred_list[0].shape[1]), dtype=np.float64)
    for w, p in zip(weights, pred_list):
        stacked += w * p.loc[common].values
    stacked /= sum(weights)
    ens = pd.DataFrame(stacked, index=common, columns=pred_list[0].columns)

    y = pd.read_csv("/root/commodity/data/train_labels.csv").set_index("date_id").loc[common]
    folds = pd.read_csv("/root/commodity/validation/fold_assignments.csv").set_index("date_id")
    print(f"Ensemble of {args.slugs} (weights {weights}, rank_norm={args.rank_normalize}):")
    for f in range(4):
        idx = folds.loc[folds.fold == f].index.intersection(common)
        rhos = per_date_spearman(ens.loc[idx], y.loc[idx])
        print(f"  fold {f}: Sharpe={sharpe_from_rhos(rhos):+.4f}  mean_rho={rhos.mean():+.4f}")
    rhos_all = per_date_spearman(ens, y)
    print(f"  OVERALL: Sharpe={sharpe_from_rhos(rhos_all):+.4f}  mean_rho={rhos_all.mean():+.4f}")


if __name__ == "__main__":
    main()
