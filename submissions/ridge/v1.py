"""v1: sanity multi-output ridge.

Features: 1-day trailing log-returns on all price-like columns (LME close,
JPX close, US adj_close, FX). To avoid fold-gap issues (training has a hole
and test has no history), features are computed from the full raw X panel
on disk — features at date d use only X values at dates <= d (causal), so
this is not leakage. Zero-impute label NaNs. Single closed-form ridge fit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
FX_PREFIX = "FX_"
PRICE_SUFFIXES = ("_Close", "adj_close")


def price_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c.startswith(FX_PREFIX):
            cols.append(c)
        elif any(s in c for s in PRICE_SUFFIXES):
            cols.append(c)
    return cols


def compute_features_full() -> pd.DataFrame:
    X_full = pd.read_csv(DATA_DIR / "train.csv").set_index("date_id").sort_index()
    prices = X_full[price_columns(X_full)].astype(np.float64)
    prices = prices.ffill()
    log_prices = np.log(prices.where(prices > 0))
    ret_1 = log_prices.diff(1)
    ret_1.columns = [f"{c}__lret1" for c in ret_1.columns]
    feats = ret_1.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feats


class Model:
    device = "cpu"

    def __init__(self) -> None:
        self.alpha = 10.0
        self.reg: Ridge | None = None
        self._target_cols: list[str] | None = None
        self._feature_cols: list[str] | None = None
        self._features: pd.DataFrame | None = None

    def _ensure_features(self) -> pd.DataFrame:
        if self._features is None:
            self._features = compute_features_full()
        return self._features

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        feats = self._ensure_features()
        train_feats = feats.loc[X.index]
        y_imp = y.fillna(0.0).astype(np.float64)
        self._target_cols = list(y_imp.columns)
        self._feature_cols = list(train_feats.columns)
        self.reg = Ridge(alpha=self.alpha, fit_intercept=True)
        self.reg.fit(train_feats.values, y_imp.values)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.reg is not None and self._target_cols is not None
        feats = self._ensure_features()
        test_feats = feats.loc[X.index].reindex(columns=self._feature_cols).fillna(0.0)
        preds = self.reg.predict(test_feats.values)
        return pd.DataFrame(preds, index=X.index, columns=self._target_cols)
