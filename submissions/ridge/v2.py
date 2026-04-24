"""v2: full classical feature set.

- log-returns at {1, 5, 20, 60}
- rolling vol at {5, 20, 60}
- rolling mean-return at {5, 20, 60}
- cross-sectional rank of 1-day returns and of 20-day returns

All causal. Features computed on full X panel. Ridge(alpha=10) on standardized features.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from utils import (
    clean,
    cross_sectional_rank,
    load_raw_X,
    log_returns,
    price_columns,
    rolling_mean_return,
    rolling_vol,
    standardize_apply,
    standardize_fit,
)


_FEATURES: pd.DataFrame | None = None


def compute_features() -> pd.DataFrame:
    global _FEATURES
    if _FEATURES is not None:
        return _FEATURES
    X = load_raw_X()
    prices = X[price_columns(X)].astype(np.float64).ffill()
    log_p = np.log(prices.where(prices > 0))

    parts = []
    parts.append(log_returns(prices, [1, 5, 20, 60]))
    parts.append(rolling_vol(log_p, [5, 20, 60]))
    parts.append(rolling_mean_return(log_p, [5, 20, 60]))

    r1 = log_p.diff(1)
    r20 = log_p.diff(20)
    parts.append(cross_sectional_rank(r1, "xsr1"))
    parts.append(cross_sectional_rank(r20, "xsr20"))

    feats = pd.concat(parts, axis=1)
    _FEATURES = clean(feats)
    return _FEATURES


class Model:
    device = "cpu"

    def __init__(self) -> None:
        self.alpha = 10.0
        self.reg: Ridge | None = None
        self._target_cols: list[str] | None = None
        self._feature_cols: list[str] | None = None
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        feats = compute_features().loc[X.index]
        y_imp = y.fillna(0.0).astype(np.float64)
        self._target_cols = list(y_imp.columns)
        self._feature_cols = list(feats.columns)
        Xs, self._mu, self._sd = standardize_fit(feats.values.astype(np.float64))
        self.reg = Ridge(alpha=self.alpha, fit_intercept=True)
        self.reg.fit(Xs, y_imp.values)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.reg is not None and self._target_cols is not None
        feats = compute_features().loc[X.index].reindex(columns=self._feature_cols)
        Xs = standardize_apply(feats.values.astype(np.float64), self._mu, self._sd)
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        preds = self.reg.predict(Xs)
        return pd.DataFrame(preds, index=X.index, columns=self._target_cols)
