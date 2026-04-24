"""Smoke-test model: predicts independent random numbers for every
(date, target) pair. Useful for verifying that the validator pipeline
runs end-to-end. Expected Sharpe is near zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class Model:
    device = "cpu"

    def __init__(self) -> None:
        self._target_cols: list[str] | None = None
        self._rng = np.random.default_rng(0)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self._target_cols = list(y.columns)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self._target_cols is not None
        preds = self._rng.standard_normal((len(X), len(self._target_cols)))
        return pd.DataFrame(preds, index=X.index, columns=self._target_cols)
