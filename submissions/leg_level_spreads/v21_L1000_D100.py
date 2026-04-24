"""v21: v19 + pair-spread features (target-specific).

Augments `wide_features` with `pair_spread_features` for the ~85 unique
pairs that appear as spread targets. Keeps the 4-way ridge ensemble.
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

from features import pair_spread_features, wide_features
from utils import DATA_DIR, LAGS, compose_targets, leg_returns, load_target_pairs, unique_legs


ALPHA_LEG = 1000.0
ALPHA_DIRECT = 100.0


def _zscore(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=0)
    sd = df.std(axis=0).replace(0, 1.0)
    return (df - mu) / sd


class Model:
    device = "cpu"

    def __init__(self) -> None:
        self.tp = load_target_pairs()
        self.legs = unique_legs(self.tp)
        self.prices = (
            pd.read_csv(DATA_DIR / "train.csv").set_index("date_id").sort_index()
        )

    def _demean_leg_per_lag(self, Y: pd.DataFrame) -> pd.DataFrame:
        out = Y.copy()
        for lag in LAGS:
            lag_cols = [c for c in Y.columns if c[1] == lag]
            sub = Y[lag_cols]
            out[lag_cols] = sub.sub(sub.mean(axis=1), axis=0)
        return out

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        F_wide = wide_features(self.prices, self.legs)
        F_pair = pair_spread_features(self.prices, self.tp)
        F = pd.concat([F_wide, F_pair], axis=1)
        self._F = F
        F_tr = F.loc[X.index].to_numpy(dtype=np.float32)
        F_tr = np.nan_to_num(F_tr, nan=0.0, posinf=0.0, neginf=0.0)

        self._scaler = StandardScaler()
        F_tr = self._scaler.fit_transform(F_tr)

        Y_leg = leg_returns(self.prices, self.legs, lags=LAGS).loc[X.index]
        self._leg_cols = list(Y_leg.columns)
        Y_leg_raw = np.nan_to_num(Y_leg.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        Y_leg_dm = np.nan_to_num(
            self._demean_leg_per_lag(Y_leg).to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )

        self._ridge_leg_raw = Ridge(alpha=ALPHA_LEG, fit_intercept=True)
        self._ridge_leg_raw.fit(F_tr, Y_leg_raw)
        self._ridge_leg_dm = Ridge(alpha=ALPHA_LEG, fit_intercept=True)
        self._ridge_leg_dm.fit(F_tr, Y_leg_dm)

        self._target_cols = list(y.columns)
        y_raw = np.nan_to_num(y.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        y_dm_np = y.to_numpy(dtype=np.float32, copy=True)
        row_mean = np.nanmean(y_dm_np, axis=1, keepdims=True)
        row_mean = np.where(np.isnan(row_mean), 0.0, row_mean)
        y_dm_np = y_dm_np - row_mean
        y_dm_np = np.nan_to_num(y_dm_np, nan=0.0, posinf=0.0, neginf=0.0)

        self._ridge_direct_raw = Ridge(alpha=ALPHA_DIRECT, fit_intercept=True)
        self._ridge_direct_raw.fit(F_tr, y_raw)
        self._ridge_direct_dm = Ridge(alpha=ALPHA_DIRECT, fit_intercept=True)
        self._ridge_direct_dm.fit(F_tr, y_dm_np)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        F_te = self._F.loc[X.index].to_numpy(dtype=np.float32)
        F_te = np.nan_to_num(F_te, nan=0.0, posinf=0.0, neginf=0.0)
        F_te = self._scaler.transform(F_te)

        def leg_to_target(arr):
            df = pd.DataFrame(
                arr,
                index=X.index,
                columns=pd.MultiIndex.from_tuples(self._leg_cols, names=["asset", "lag"]),
            )
            return compose_targets(df, self.tp)

        leg_raw = leg_to_target(self._ridge_leg_raw.predict(F_te))
        leg_dm = leg_to_target(self._ridge_leg_dm.predict(F_te))
        direct_raw = pd.DataFrame(
            self._ridge_direct_raw.predict(F_te),
            index=X.index, columns=self._target_cols,
        )[leg_raw.columns]
        direct_dm = pd.DataFrame(
            self._ridge_direct_dm.predict(F_te),
            index=X.index, columns=self._target_cols,
        )[leg_raw.columns]

        return (
            _zscore(leg_raw) + _zscore(leg_dm) + _zscore(direct_raw) + _zscore(direct_dm)
        ) / 4.0
