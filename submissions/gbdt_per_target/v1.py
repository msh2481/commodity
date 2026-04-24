"""v1: minimal pipeline sanity.

- Features: 4 LME close prices + their 1-day log returns (8 features).
- XGBoost one-per-target on GPU, 100 trees, max_depth=4, defaults otherwise.
- Fills NaN features with ffill+0; drops NaN target rows per-target when fitting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb


FEATURE_COLS = ["LME_AH_Close", "LME_CA_Close", "LME_PB_Close", "LME_ZS_Close"]


def build_features(X: pd.DataFrame) -> pd.DataFrame:
    """Ffill prices then compute 1-day log returns. Index preserved."""
    Xs = X.sort_index()
    prices = Xs[FEATURE_COLS].ffill()
    logp = np.log(prices.clip(lower=1e-9))
    ret1 = logp.diff().add_suffix("_ret1")
    feats = pd.concat([prices, ret1], axis=1).fillna(0.0).astype(np.float32)
    return feats


class Model:
    device = "cuda"

    def __init__(self) -> None:
        self._target_cols: list[str] | None = None
        self._models: dict[str, xgb.XGBRegressor] = {}

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self._target_cols = list(y.columns)
        feats = build_features(X)
        feats_aligned = feats.loc[y.index]
        X_np = feats_aligned.to_numpy()
        for tgt in self._target_cols:
            yt = y[tgt].values
            mask = ~np.isnan(yt)
            if mask.sum() < 50:
                self._models[tgt] = None
                continue
            m = xgb.XGBRegressor(
                device="cuda",
                tree_method="hist",
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                verbosity=0,
            )
            m.fit(X_np[mask], yt[mask].astype(np.float32))
            self._models[tgt] = m

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self._target_cols is not None
        feats = build_features(X)
        X_np = feats.loc[X.index].to_numpy()
        out = np.zeros((len(X), len(self._target_cols)), dtype=np.float32)
        for j, tgt in enumerate(self._target_cols):
            m = self._models.get(tgt)
            if m is None:
                continue
            out[:, j] = m.predict(X_np)
        return pd.DataFrame(out, index=X.index, columns=self._target_cols)
