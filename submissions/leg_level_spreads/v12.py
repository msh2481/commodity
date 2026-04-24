"""v12: v7 + per-lag alpha tuning for the leg model.

Fits 4 separate leg ridges (one per lag) with possibly different alpha,
then composes targets. Keep direct ridge and the z-score ensemble.
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


ALPHA_PER_LAG = {1: 30.0, 2: 50.0, 3: 70.0, 4: 100.0}
ALPHA_DIRECT = 10.0


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

        # Split by lag
        self._leg_ridges = {}
        self._lag_col_idx = {}
        for lag in LAGS:
            mask = np.array([c[1] == lag for c in self._leg_cols])
            self._lag_col_idx[lag] = mask
            Y_lag = Y_leg_np[:, mask]
            r = Ridge(alpha=ALPHA_PER_LAG[lag], fit_intercept=True)
            r.fit(F_tr, Y_lag)
            self._leg_ridges[lag] = r

        self._target_cols = list(y.columns)
        Y_direct = np.nan_to_num(y.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        self._ridge_direct = Ridge(alpha=ALPHA_DIRECT, fit_intercept=True)
        self._ridge_direct.fit(F_tr, Y_direct)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        F_te = self._F.loc[X.index].to_numpy(dtype=np.float32)
        F_te = np.nan_to_num(F_te, nan=0.0, posinf=0.0, neginf=0.0)
        F_te = self._scaler.transform(F_te)

        # stitch per-lag predictions back into full leg matrix
        full_pred = np.empty((F_te.shape[0], len(self._leg_cols)), dtype=np.float32)
        for lag in LAGS:
            idx = self._lag_col_idx[lag]
            full_pred[:, idx] = self._leg_ridges[lag].predict(F_te)

        leg_df = pd.DataFrame(
            full_pred,
            index=X.index,
            columns=pd.MultiIndex.from_tuples(self._leg_cols, names=["asset", "lag"]),
        )
        leg_target = compose_targets(leg_df, self.tp)

        direct_pred = self._ridge_direct.predict(F_te)
        direct_df = pd.DataFrame(direct_pred, index=X.index, columns=self._target_cols)
        direct_df = direct_df[leg_target.columns]

        def zscore(df: pd.DataFrame) -> pd.DataFrame:
            mu = df.mean(axis=0)
            sd = df.std(axis=0).replace(0, 1.0)
            return (df - mu) / sd

        return 0.5 * (zscore(leg_target) + zscore(direct_df))
