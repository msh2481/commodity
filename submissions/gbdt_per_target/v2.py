"""v2: full feature set baseline (APPROACH step 2).

Features per asset (143 price-like cols):
  - log-returns at 1, 5, 20 day lags (3)
  - 20d rolling vol of 1-day log returns (1)
  - 20d rolling mean of 1-day log returns (1)
  - cross-sectional pct rank of 1-day return (1)
Plus 3 calendar features (dow, wom, trend).

XGBoost one-per-target on GPU:
  - n_estimators=500, max_depth=6, learning_rate=0.05
  - min_child_weight=5, subsample=0.8, colsample_bytree=0.7
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import xgboost as xgb

from utils import build_feature_matrix


class Model:
    device = "cuda"

    def __init__(self) -> None:
        self._target_cols: list[str] | None = None
        self._models: dict[str, xgb.XGBRegressor] = {}

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self._target_cols = list(y.columns)
        feats = build_feature_matrix(X)
        feats_aligned = feats.loc[y.index].to_numpy()

        for tgt in self._target_cols:
            yt = y[tgt].values
            mask = ~np.isnan(yt)
            if mask.sum() < 50:
                self._models[tgt] = None
                continue
            m = xgb.XGBRegressor(
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
                n_jobs=4,
            )
            m.fit(feats_aligned[mask], yt[mask].astype(np.float32))
            self._models[tgt] = m

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self._target_cols is not None
        feats = build_feature_matrix(X)
        X_np = feats.loc[X.index].to_numpy()
        out = np.zeros((len(X), len(self._target_cols)), dtype=np.float32)
        for j, tgt in enumerate(self._target_cols):
            m = self._models.get(tgt)
            if m is None:
                continue
            out[:, j] = m.predict(X_np)
        return pd.DataFrame(out, index=X.index, columns=self._target_cols)
