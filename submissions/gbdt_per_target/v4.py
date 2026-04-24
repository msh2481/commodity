"""v4: v2 + per-target prediction standardization.

Same features & hparams as v2. After fitting, compute each target's
train-set prediction mean and std; at predict time, z-score predictions
by (pred - mean) / std so each target contributes on a comparable scale
to the cross-sectional ranking.
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
        self._means: dict[str, float] = {}
        self._stds: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self._target_cols = list(y.columns)
        feats = build_feature_matrix(X)
        feats_aligned = feats.loc[y.index].to_numpy()

        for tgt in self._target_cols:
            yt = y[tgt].values
            mask = ~np.isnan(yt)
            if mask.sum() < 50:
                self._models[tgt] = None
                self._means[tgt] = 0.0
                self._stds[tgt] = 1.0
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
            # Train-set prediction stats for later standardization
            tr_pred = m.predict(feats_aligned[mask])
            self._means[tgt] = float(np.mean(tr_pred))
            s = float(np.std(tr_pred))
            self._stds[tgt] = s if s > 1e-9 else 1.0

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self._target_cols is not None
        feats = build_feature_matrix(X)
        X_np = feats.loc[X.index].to_numpy()
        out = np.zeros((len(X), len(self._target_cols)), dtype=np.float32)
        for j, tgt in enumerate(self._target_cols):
            m = self._models.get(tgt)
            if m is None:
                continue
            raw = m.predict(X_np)
            out[:, j] = (raw - self._means[tgt]) / self._stds[tgt]
        return pd.DataFrame(out, index=X.index, columns=self._target_cols)
