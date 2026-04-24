"""v4: direct per-target ridge (within-thread sanity baseline).

Uses the same extended feature set as v3, but fits ridge directly
against the 424 target columns from y (no leg composition). If the
compositional hypothesis is right, v3 (leg model) should match or beat
v4 on overall Sharpe.
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
from sklearn.preprocessing import StandardScaler

from features import extended_features
from utils import DATA_DIR, load_target_pairs, unique_legs


class Model:
    device = "cpu"

    def __init__(self) -> None:
        self.tp = load_target_pairs()
        self.legs = unique_legs(self.tp)
        self.prices = (
            pd.read_csv(DATA_DIR / "train.csv").set_index("date_id").sort_index()
        )
        self._scaler: StandardScaler | None = None
        self._ridge: Ridge | None = None
        self._target_cols: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        F = extended_features(self.prices, self.legs)

        F_tr = F.loc[X.index].to_numpy(dtype=np.float32)
        F_tr = np.nan_to_num(F_tr, nan=0.0, posinf=0.0, neginf=0.0)

        self._target_cols = list(y.columns)
        Y_tr = y.to_numpy(dtype=np.float32)
        Y_tr = np.nan_to_num(Y_tr, nan=0.0, posinf=0.0, neginf=0.0)

        self._scaler = StandardScaler()
        F_tr = self._scaler.fit_transform(F_tr)

        self._ridge = Ridge(alpha=10.0, fit_intercept=True)
        self._ridge.fit(F_tr, Y_tr)
        self._F = F

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self._scaler is not None and self._ridge is not None
        assert self._target_cols is not None
        F_te = self._F.loc[X.index].to_numpy(dtype=np.float32)
        F_te = np.nan_to_num(F_te, nan=0.0, posinf=0.0, neginf=0.0)
        F_te = self._scaler.transform(F_te)
        preds = self._ridge.predict(F_te)
        return pd.DataFrame(preds, index=X.index, columns=self._target_cols)
