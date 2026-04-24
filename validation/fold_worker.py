"""Run a single validation fold: fit a Model on the training dates and
predict on the held-out fold. Writes predictions to --out as a CSV with
columns date_id, target_0, ..., target_423.

Invoked by pueue from validate.py. Independent process: seeds are set
before any user code runs, and CUDA_VISIBLE_DEVICES must already be set
in the environment (via `env` wrapper on the pueue command) if GPU is
wanted.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42


def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_model_class(model_path: Path):
    # Allow the user's model file to import sibling modules from its own
    # directory (e.g. `submissions/<thread>/utils.py`).
    parent = str(model_path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location("user_model", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "Model"):
        raise RuntimeError(f"{model_path} must define a class named `Model`")
    return module.Model


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--folds", type=Path, required=True,
                   help="fold_assignments.csv path")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    set_seeds(SEED)

    folds = pd.read_csv(args.folds)
    test_dates = set(folds.loc[folds.fold == args.fold, "date_id"])
    train_dates = set(folds.loc[folds.fold != args.fold, "date_id"])
    if not test_dates:
        raise RuntimeError(f"No dates assigned to fold {args.fold}")

    X_full = pd.read_csv(args.data_dir / "train.csv").set_index("date_id")
    y_full = pd.read_csv(args.data_dir / "train_labels.csv").set_index("date_id")
    target_cols = list(y_full.columns)

    X_train = X_full.loc[X_full.index.isin(train_dates)].sort_index()
    y_train = y_full.loc[y_full.index.isin(train_dates)].sort_index()
    X_test = X_full.loc[X_full.index.isin(test_dates)].sort_index()

    print(f"[fold {args.fold}] train={len(X_train)} test={len(X_test)}", flush=True)

    Model = load_model_class(args.model)
    model = Model()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    # also predict on the training dates for train/test gap diagnostics
    try:
        train_preds = model.predict(X_train)
    except Exception as e:
        print(f"[fold {args.fold}] train-pred failed: {e}", file=sys.stderr, flush=True)
        train_preds = None

    if not isinstance(preds, pd.DataFrame):
        raise RuntimeError("Model.predict must return a DataFrame")
    if preds.index.name != "date_id":
        raise RuntimeError("predictions DataFrame must be indexed by date_id")

    missing = [c for c in target_cols if c not in preds.columns]
    if missing:
        raise RuntimeError(f"predictions missing columns: {missing[:5]}... (total {len(missing)})")
    preds = preds[target_cols]

    # Align to the expected test dates; fill missing rows with 0.
    expected_dates = sorted(test_dates)
    n_missing_rows = sum(1 for d in expected_dates if d not in preds.index)
    preds = preds.reindex(expected_dates)

    n_nan = int(preds.isna().sum().sum())
    if n_nan or n_missing_rows:
        print(
            f"[fold {args.fold}] WARNING: {n_nan} NaN cells, "
            f"{n_missing_rows} missing rows — filling with 0",
            file=sys.stderr,
            flush=True,
        )
    preds = preds.fillna(0.0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    preds.reset_index().to_csv(args.out, index=False)
    print(f"[fold {args.fold}] wrote {args.out} ({preds.shape[0]} rows)", flush=True)

    if isinstance(train_preds, pd.DataFrame) and train_preds.index.name == "date_id":
        missing_tr = [c for c in target_cols if c not in train_preds.columns]
        if not missing_tr:
            train_preds = train_preds[target_cols]
            train_preds = train_preds.reindex(sorted(train_dates)).fillna(0.0)
            train_out = args.out.with_name(args.out.stem + "_train.csv")
            train_preds.reset_index().to_csv(train_out, index=False)
            print(f"[fold {args.fold}] wrote {train_out} ({train_preds.shape[0]} rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
