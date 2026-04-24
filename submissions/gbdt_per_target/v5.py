"""v5: v2 hparams + thread-parallel XGB fits for speed.

Same features and hparams as v2 (baseline). Only the fit loop is
parallelized with a thread pool — XGBoost CUDA kernels overlap and
give ~2-3x throughput on a single GPU.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import concurrent.futures
import numpy as np
import pandas as pd
import xgboost as xgb

from utils import build_feature_matrix


N_THREADS_FIT = 8


def _fit_one(args):
    X_np, yt, params = args
    mask = ~np.isnan(yt)
    if mask.sum() < 50:
        return None
    m = xgb.XGBRegressor(**params)
    m.fit(X_np[mask], yt[mask].astype(np.float32))
    return m


class Model:
    device = "cuda"

    def __init__(self) -> None:
        self._target_cols: list[str] | None = None
        self._models: list[xgb.XGBRegressor | None] = []

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self._target_cols = list(y.columns)
        feats = build_feature_matrix(X)
        X_np = feats.loc[y.index].to_numpy()

        params = dict(
            device="cuda",
            tree_method="hist",
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_lambda=1.0,
            verbosity=0,
            n_jobs=1,
        )
        jobs = [(X_np, y[t].values, params) for t in self._target_cols]
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
            self._models = list(ex.map(_fit_one, jobs))

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self._target_cols is not None
        feats = build_feature_matrix(X)
        X_np = feats.loc[X.index].to_numpy()
        out = np.zeros((len(X), len(self._target_cols)), dtype=np.float32)

        def _pred(args):
            j, m = args
            if m is None:
                return j, None
            return j, m.predict(X_np)

        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
            for j, p in ex.map(_pred, enumerate(self._models)):
                if p is not None:
                    out[:, j] = p
        return pd.DataFrame(out, index=X.index, columns=self._target_cols)
