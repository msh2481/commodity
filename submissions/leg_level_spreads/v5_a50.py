"""v5: ridge leg model, wide features (all price-like cols in train.csv).

Per-column ret1/5/20 across ALL 450 price-like columns (not just legs),
plus leg-only ret60/vol20/mom20/xrank5/beta20, plus bucket-factor
returns and calendar. Alpha tuned to 500 (midway between v3 sweep peaks
at 100 and 1000).
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

from features import wide_features
from utils import DATA_DIR, LAGS, compose_targets, leg_returns, load_target_pairs, unique_legs


ALPHA = 50.0


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
        self._leg_cols: list[tuple[str, int]] | None = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        F = wide_features(self.prices, self.legs)
        Y = leg_returns(self.prices, self.legs, lags=LAGS)

        F_tr = F.loc[X.index].to_numpy(dtype=np.float32)
        Y_tr = Y.loc[X.index]
        self._leg_cols = list(Y_tr.columns)
        Y_tr = Y_tr.to_numpy(dtype=np.float32)

        F_tr = np.nan_to_num(F_tr, nan=0.0, posinf=0.0, neginf=0.0)
        Y_tr = np.nan_to_num(Y_tr, nan=0.0, posinf=0.0, neginf=0.0)

        self._scaler = StandardScaler()
        F_tr = self._scaler.fit_transform(F_tr)

        self._ridge = Ridge(alpha=ALPHA, fit_intercept=True)
        self._ridge.fit(F_tr, Y_tr)
        self._F = F

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self._scaler is not None and self._ridge is not None
        assert self._leg_cols is not None
        F_te = self._F.loc[X.index].to_numpy(dtype=np.float32)
        F_te = np.nan_to_num(F_te, nan=0.0, posinf=0.0, neginf=0.0)
        F_te = self._scaler.transform(F_te)
        preds = self._ridge.predict(F_te)

        leg_preds = pd.DataFrame(
            preds,
            index=X.index,
            columns=pd.MultiIndex.from_tuples(self._leg_cols, names=["asset", "lag"]),
        )
        return compose_targets(leg_preds, self.tp)
