"""v3: regularized (APPROACH step 3).

Same features as v2. Heavier regularization:
  - max_depth=4 (down from 6)
  - min_child_weight=20 (up from 5)
  - reg_lambda=5.0 (up from 1.0)
  - subsample=0.7, colsample_bytree=0.5
  - gamma=0.1
  - n_estimators=300 (fewer trees since deeper reg)
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
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                min_child_weight=20,
                subsample=0.7,
                colsample_bytree=0.5,
                reg_lambda=5.0,
                gamma=0.1,
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
