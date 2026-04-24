"""v16: v7 with per-date cross-sectional demeaning of leg labels.

The metric evaluates cross-sectional Spearman per date — only relative
ordering across targets matters. Demean leg-return labels per (date,
lag) across the 106 legs, so the ridge fit concentrates on *relative*
movements instead of learning the grand mean return per date.
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


class Model:
    device = "cpu"

    def __init__(self) -> None:
        self.tp = load_target_pairs()
        self.legs = unique_legs(self.tp)
        self.prices = (
            pd.read_csv(DATA_DIR / "train.csv").set_index("date_id").sort_index()
        )

    def _demean_per_date_per_lag(self, Y: pd.DataFrame) -> pd.DataFrame:
        """Y has MultiIndex cols (asset, lag). Subtract mean over assets
        within each date, separately for each lag."""
        out = Y.copy()
        for lag in LAGS:
            lag_cols = [c for c in Y.columns if c[1] == lag]
            sub = Y[lag_cols]
            row_mean = sub.mean(axis=1)
            out[lag_cols] = sub.sub(row_mean, axis=0)
        return out

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        F = wide_features(self.prices, self.legs)
        self._F = F
        F_tr = F.loc[X.index].to_numpy(dtype=np.float32)
        F_tr = np.nan_to_num(F_tr, nan=0.0, posinf=0.0, neginf=0.0)

        self._scaler = StandardScaler()
        F_tr = self._scaler.fit_transform(F_tr)

        Y_leg = leg_returns(self.prices, self.legs, lags=LAGS).loc[X.index]
        self._leg_cols = list(Y_leg.columns)
        # cross-section demean per (date, lag)
        Y_leg_dm = self._demean_per_date_per_lag(Y_leg)
        Y_leg_np = np.nan_to_num(
            Y_leg_dm.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )

        self._ridge_leg = Ridge(alpha=ALPHA_LEG, fit_intercept=True)
        self._ridge_leg.fit(F_tr, Y_leg_np)

        self._target_cols = list(y.columns)
        # for direct ridge, demean across targets per date
        y_np = y.to_numpy(dtype=np.float32, copy=True)
        # row-wise mean excluding NaN, then subtract
        y_mean = np.nanmean(y_np, axis=1, keepdims=True)
        y_mean = np.where(np.isnan(y_mean), 0.0, y_mean)
        y_np = y_np - y_mean
        y_np = np.nan_to_num(y_np, nan=0.0, posinf=0.0, neginf=0.0)
        self._ridge_direct = Ridge(alpha=ALPHA_DIRECT, fit_intercept=True)
        self._ridge_direct.fit(F_tr, y_np)

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

        def zscore(df: pd.DataFrame) -> pd.DataFrame:
            mu = df.mean(axis=0)
            sd = df.std(axis=0).replace(0, 1.0)
            return (df - mu) / sd

        return 0.5 * (zscore(leg_target) + zscore(direct_df))
