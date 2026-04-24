"""v8: ensemble with per-date rank averaging.

Same two models as v7 (leg-ridge + direct-ridge), but combines them by
converting each target column to cross-sectional ranks (within date)
then averaging ranks. This is even closer to what the metric scores.
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


ALPHA_LEG = 30.0
ALPHA_DIRECT = 10.0
WEIGHT_LEG = 0.5


class Model:
    device = "cpu"

    def __init__(self) -> None:
        self.tp = load_target_pairs()
        self.legs = unique_legs(self.tp)
        self.prices = (
            pd.read_csv(DATA_DIR / "train.csv").set_index("date_id").sort_index()
        )

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        F = wide_features(self.prices, self.legs)
        self._F = F
        F_tr = F.loc[X.index].to_numpy(dtype=np.float32)
        F_tr = np.nan_to_num(F_tr, nan=0.0, posinf=0.0, neginf=0.0)

        self._scaler = StandardScaler()
        F_tr = self._scaler.fit_transform(F_tr)

        Y_leg = leg_returns(self.prices, self.legs, lags=LAGS).loc[X.index]
        self._leg_cols = list(Y_leg.columns)
        Y_leg_np = np.nan_to_num(Y_leg.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        self._ridge_leg = Ridge(alpha=ALPHA_LEG, fit_intercept=True)
        self._ridge_leg.fit(F_tr, Y_leg_np)

        self._target_cols = list(y.columns)
        Y_direct = np.nan_to_num(y.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        self._ridge_direct = Ridge(alpha=ALPHA_DIRECT, fit_intercept=True)
        self._ridge_direct.fit(F_tr, Y_direct)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        F_te = self._F.loc[X.index].to_numpy(dtype=np.float32)
        F_te = np.nan_to_num(F_te, nan=0.0, posinf=0.0, neginf=0.0)
        F_te = self._scaler.transform(F_te)

        leg_pred = self._ridge_leg.predict(F_te)
        direct_pred = self._ridge_direct.predict(F_te)

        leg_df = pd.DataFrame(
            leg_pred,
            index=X.index,
            columns=pd.MultiIndex.from_tuples(self._leg_cols, names=["asset", "lag"]),
        )
        leg_target = compose_targets(leg_df, self.tp)
        direct_df = pd.DataFrame(direct_pred, index=X.index, columns=self._target_cols)
        direct_df = direct_df[leg_target.columns]

        # cross-sectional rank per date (axis=1), scaled to [0, 1]
        r_leg = leg_target.rank(axis=1, pct=True)
        r_dir = direct_df.rank(axis=1, pct=True)
        return WEIGHT_LEG * r_leg + (1 - WEIGHT_LEG) * r_dir
