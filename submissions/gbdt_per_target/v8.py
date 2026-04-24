"""v8: v1 hparams + v2 features. Isolate feature-set contribution.

Uses v1 hparams (100 trees, depth 4, lr 0.05, min_child_weight 5) but
v2's full feature set (861 features). Comparing v8 vs v1 vs v2:
  - v8 > v2: v2's big hparams overfit; smaller hparams fix it
  - v8 > v1: features help when hparams are conservative
  - v8 ≈ v1: features at conservative hparams don't improve
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


def _predict_one(args):
    X_np, m = args
    if m is None:
        return None
    return m.predict(X_np)


class Model:
    device = "cuda"

    def __init__(self) -> None:
        self._target_cols: list[str] | None = None
        self._models: list[xgb.XGBRegressor | None] = []
        self._X_train: pd.DataFrame | None = None

    def _combine_X(self, test_X: pd.DataFrame | None = None) -> pd.DataFrame:
        if test_X is None:
            return self._X_train.sort_index()
        combined = pd.concat([self._X_train, test_X])
        combined = combined[~combined.index.duplicated(keep="first")].sort_index()
        return combined

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self._target_cols = list(y.columns)
        self._X_train = X.copy()
        full_X = self._combine_X(None)
        feats = build_feature_matrix(full_X).loc[y.index]
        X_np = feats.to_numpy()

        params = dict(
            device="cuda",
            tree_method="hist",
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            verbosity=0,
            n_jobs=1,
        )
        jobs = [(X_np, y[t].values, params) for t in self._target_cols]
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
            self._models = list(ex.map(_fit_one, jobs))

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self._target_cols is not None
        full_X = self._combine_X(X)
        feats = build_feature_matrix(full_X).loc[X.index]
        X_np = feats.to_numpy()
        jobs = [(X_np, m) for m in self._models]
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
            results = list(ex.map(_predict_one, jobs))
        out = np.zeros((len(X), len(self._target_cols)), dtype=np.float32)
        for j, p in enumerate(results):
            if p is not None:
                out[:, j] = p
        return pd.DataFrame(out, index=X.index, columns=self._target_cols)
