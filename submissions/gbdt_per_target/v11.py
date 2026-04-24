"""v11: v1 hparams, expand to slightly larger LME-centric feature set.

Tests: how much of v1's edge comes from its exact 8 features, vs. a
small LME-centric set generally.

Features (12):
  - 4 LME closes (raw)
  - 4 LME 1-day log-returns
  - 4 LME 5-day log-returns
v1-like hparams: 100 trees, depth 4, lr 0.05, mcw 5.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import concurrent.futures
import numpy as np
import pandas as pd
import xgboost as xgb


LME_COLS = ["LME_AH_Close", "LME_CA_Close", "LME_PB_Close", "LME_ZS_Close"]
N_THREADS_FIT = 8


def build_feats(X: pd.DataFrame) -> np.ndarray:
    Xs = X.sort_index()
    prices = Xs[LME_COLS].ffill()
    logp = np.log(prices.clip(lower=1e-9))
    ret1 = logp.diff()
    ret5 = logp - logp.shift(5)
    ret20 = logp - logp.shift(20)
    feats = pd.concat(
        [prices, ret1.add_suffix("_ret1"), ret5.add_suffix("_ret5"), ret20.add_suffix("_ret20")],
        axis=1,
    ).fillna(0.0).astype(np.float32)
    return feats.values


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

    def _combine(self, test_X: pd.DataFrame | None) -> pd.DataFrame:
        if test_X is None:
            return self._X_train.sort_index()
        c = pd.concat([self._X_train, test_X])
        return c[~c.index.duplicated(keep="first")].sort_index()

    def fit(self, X, y):
        self._target_cols = list(y.columns)
        self._X_train = X.copy()
        full_X = self._combine(None)
        feats = build_feats(full_X)
        idx = full_X.index
        pos = idx.get_indexer(y.index)
        X_np = feats[pos]
        params = dict(device="cuda", tree_method="hist", n_estimators=100,
                      max_depth=4, learning_rate=0.05, verbosity=0, n_jobs=1)
        jobs = [(X_np, y[t].values, params) for t in self._target_cols]
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
            self._models = list(ex.map(_fit_one, jobs))

    def predict(self, X):
        full_X = self._combine(X)
        feats = build_feats(full_X)
        pos = full_X.index.get_indexer(X.index)
        X_np = feats[pos]
        jobs = [(X_np, m) for m in self._models]
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
            results = list(ex.map(_predict_one, jobs))
        out = np.zeros((len(X), len(self._target_cols)), dtype=np.float32)
        for j, p in enumerate(results):
            if p is not None:
                out[:, j] = p
        return pd.DataFrame(out, index=X.index, columns=self._target_cols)
