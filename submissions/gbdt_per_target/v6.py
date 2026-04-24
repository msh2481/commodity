"""v6: v2 features + per-target spread-specific features, thread-parallel.

For each spread target 'A - B', add 3 pair-specific features:
  - log(A/B) current spread
  - deviation from 20d SMA of the spread
  - z-score of the spread over 20d window

Also fix fold-boundary NaN: features are computed on the concat of
(stored train X) + (current test X) so rolling windows have context
from training data at the beginning of the test fold.

Hparams match v2 (depth 6, 500 trees, lr 0.05). Thread-parallel fits.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import concurrent.futures
import numpy as np
import pandas as pd
import xgboost as xgb

from utils import build_feature_matrix, price_columns


TARGET_PAIRS_CSV = "/root/commodity/data/target_pairs.csv"
N_THREADS_FIT = 8


def _compute_spread_features(
    X: pd.DataFrame, unique_pairs: list[tuple[str, str]]
) -> dict[tuple[str, str], pd.DataFrame]:
    """Return dict pair -> DataFrame(index=X.index, 3 cols)."""
    Xs = X.sort_index()
    prices = Xs[price_columns(Xs)].astype(np.float64).ffill()
    logp = np.log(prices.clip(lower=1e-12))
    out = {}
    for a, b in unique_pairs:
        if a not in logp.columns or b not in logp.columns:
            out[(a, b)] = None
            continue
        spread = logp[a] - logp[b]
        sma = spread.rolling(20, min_periods=10).mean()
        std = spread.rolling(20, min_periods=10).std()
        dev = spread - sma
        z = dev / std.replace(0, np.nan)
        df = pd.DataFrame({"spread": spread, "dev20": dev, "z20": z}).astype(np.float32)
        out[(a, b)] = df
    return out


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
        self._pair_for_target: list[tuple[str, str] | None] = []
        self._pairs_df = pd.read_csv(TARGET_PAIRS_CSV)
        self._X_train: pd.DataFrame | None = None

    def _pair_of(self, target: str) -> tuple[str, str] | None:
        p = self._pairs_df.loc[self._pairs_df.target == target, "pair"].iloc[0]
        if " - " in p:
            a, b = p.split(" - ")
            return a.strip(), b.strip()
        return None

    def _combine_X(self, test_X: pd.DataFrame | None = None) -> pd.DataFrame:
        if test_X is None:
            return self._X_train.sort_index()
        combined = pd.concat([self._X_train, test_X])
        combined = combined[~combined.index.duplicated(keep="first")].sort_index()
        return combined

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self._target_cols = list(y.columns)
        self._pair_for_target = [self._pair_of(t) for t in self._target_cols]
        self._X_train = X.copy()

        full_X = self._combine_X(None)
        feats = build_feature_matrix(full_X).loc[y.index]
        unique_pairs = list({p for p in self._pair_for_target if p is not None})
        spread_map = _compute_spread_features(full_X, unique_pairs)
        spread_map_sliced = {
            k: (v.loc[y.index].to_numpy() if v is not None else None)
            for k, v in spread_map.items()
        }
        generic = feats.to_numpy()

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

        jobs = []
        for j, tgt in enumerate(self._target_cols):
            pair = self._pair_for_target[j]
            spread = spread_map_sliced.get(pair) if pair is not None else None
            X_j = np.hstack([generic, spread]) if spread is not None else generic
            jobs.append((X_j, y[tgt].values, params))

        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
            self._models = list(ex.map(_fit_one, jobs))

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self._target_cols is not None
        full_X = self._combine_X(X)
        feats = build_feature_matrix(full_X).loc[X.index]
        unique_pairs = list({p for p in self._pair_for_target if p is not None})
        spread_map = _compute_spread_features(full_X, unique_pairs)
        spread_map_sliced = {
            k: (v.loc[X.index].to_numpy() if v is not None else None)
            for k, v in spread_map.items()
        }
        generic = feats.to_numpy()

        jobs = []
        for j, tgt in enumerate(self._target_cols):
            pair = self._pair_for_target[j]
            spread = spread_map_sliced.get(pair) if pair is not None else None
            X_j = np.hstack([generic, spread]) if spread is not None else generic
            jobs.append((X_j, self._models[j]))

        out = np.zeros((len(X), len(self._target_cols)), dtype=np.float32)
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
            results = list(ex.map(_predict_one, jobs))
        for j, p in enumerate(results):
            if p is not None:
                out[:, j] = p
        return pd.DataFrame(out, index=X.index, columns=self._target_cols)
