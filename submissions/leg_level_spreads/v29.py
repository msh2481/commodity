"""v28: v26 + xgb_direct_dm (XGBoost on demeaned direct targets).

8-way ensemble: 4 ridges (leg×2 + direct×2) + 4 XGBoosts (direct_raw,
direct_dm, leg_raw, leg_dm), all using the wide+pair feature set where
possible.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from features import pair_spread_features, wide_features
from utils import DATA_DIR, LAGS, compose_targets, leg_returns, load_target_pairs, unique_legs


ALPHA_LEG = 30.0
ALPHA_DIRECT = 100.0

XGB_DIRECT_PARAMS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "device": "cuda",
    "multi_strategy": "one_output_per_tree",
    "max_depth": 5,
    "eta": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.4,
    "reg_lambda": 100.0,
    "reg_alpha": 1.0,
    "verbosity": 0,
}
XGB_DIRECT_ROUNDS = 120

XGB_LEG_PARAMS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "device": "cuda",
    "multi_strategy": "one_output_per_tree",
    "max_depth": 4,
    "eta": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.4,
    "reg_lambda": 30.0,
    "verbosity": 0,
}
XGB_LEG_ROUNDS = 150


def _zscore(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=0)
    sd = df.std(axis=0).replace(0, 1.0)
    return (df - mu) / sd


class Model:
    device = "cuda"

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
        F_leg = wide_features(self.prices, self.legs)
        F_pair = pair_spread_features(self.prices, self.tp)
        F_direct = pd.concat([F_leg, F_pair], axis=1)
        self._F_leg = F_leg
        self._F_direct = F_direct

        F_leg_tr = np.nan_to_num(F_leg.loc[X.index].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        F_direct_tr = np.nan_to_num(F_direct.loc[X.index].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        self._scaler_leg = StandardScaler().fit(F_leg_tr)
        F_leg_tr_sc = self._scaler_leg.transform(F_leg_tr)
        self._scaler_direct = StandardScaler().fit(F_direct_tr)
        F_direct_tr_sc = self._scaler_direct.transform(F_direct_tr)

        Y_leg = leg_returns(self.prices, self.legs, lags=LAGS).loc[X.index]
        self._leg_cols = list(Y_leg.columns)
        Y_leg_raw = np.nan_to_num(Y_leg.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        Y_leg_dm = np.nan_to_num(
            self._demean_leg_per_lag(Y_leg).to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )

        self._ridge_leg_raw = Ridge(alpha=ALPHA_LEG, fit_intercept=True).fit(F_leg_tr_sc, Y_leg_raw)
        self._ridge_leg_dm = Ridge(alpha=ALPHA_LEG, fit_intercept=True).fit(F_leg_tr_sc, Y_leg_dm)

        self._target_cols = list(y.columns)
        y_raw = np.nan_to_num(y.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        y_dm_np = y.to_numpy(dtype=np.float32, copy=True)
        row_mean = np.nanmean(y_dm_np, axis=1, keepdims=True)
        row_mean = np.where(np.isnan(row_mean), 0.0, row_mean)
        y_dm_np = y_dm_np - row_mean
        y_dm_np = np.nan_to_num(y_dm_np, nan=0.0, posinf=0.0, neginf=0.0)

        self._ridge_direct_raw = Ridge(alpha=ALPHA_DIRECT, fit_intercept=True).fit(F_direct_tr_sc, y_raw)
        self._ridge_direct_dm = Ridge(alpha=ALPHA_DIRECT, fit_intercept=True).fit(F_direct_tr_sc, y_dm_np)

        # XGBoost direct raw
        dtrain_direct = xgb.DMatrix(F_direct_tr.astype(np.float32), label=y_raw)
        self._xgb_direct_raw = xgb.train(XGB_DIRECT_PARAMS, dtrain_direct, num_boost_round=XGB_DIRECT_ROUNDS)

        # XGBoost direct demeaned
        dtrain_direct_dm = xgb.DMatrix(F_direct_tr.astype(np.float32), label=y_dm_np)
        self._xgb_direct_dm = xgb.train(XGB_DIRECT_PARAMS, dtrain_direct_dm, num_boost_round=XGB_DIRECT_ROUNDS)

        # XGBoost leg raw
        dtrain_leg_raw = xgb.DMatrix(F_direct_tr.astype(np.float32), label=Y_leg_raw)
        self._xgb_leg_raw = xgb.train(XGB_LEG_PARAMS, dtrain_leg_raw, num_boost_round=XGB_LEG_ROUNDS)

        # XGBoost leg demeaned
        dtrain_leg_dm = xgb.DMatrix(F_direct_tr.astype(np.float32), label=Y_leg_dm)
        self._xgb_leg_dm = xgb.train(XGB_LEG_PARAMS, dtrain_leg_dm, num_boost_round=XGB_LEG_ROUNDS)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        F_leg_te = np.nan_to_num(
            self._F_leg.loc[X.index].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        F_leg_te_sc = self._scaler_leg.transform(F_leg_te)
        F_direct_te = np.nan_to_num(
            self._F_direct.loc[X.index].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        F_direct_te_sc = self._scaler_direct.transform(F_direct_te)

        def leg_to_target(arr):
            df = pd.DataFrame(
                arr,
                index=X.index,
                columns=pd.MultiIndex.from_tuples(self._leg_cols, names=["asset", "lag"]),
            )
            return compose_targets(df, self.tp)

        leg_raw = leg_to_target(self._ridge_leg_raw.predict(F_leg_te_sc))
        leg_dm = leg_to_target(self._ridge_leg_dm.predict(F_leg_te_sc))
        direct_raw = pd.DataFrame(
            self._ridge_direct_raw.predict(F_direct_te_sc),
            index=X.index, columns=self._target_cols,
        )[leg_raw.columns]
        direct_dm = pd.DataFrame(
            self._ridge_direct_dm.predict(F_direct_te_sc),
            index=X.index, columns=self._target_cols,
        )[leg_raw.columns]

        dtest = xgb.DMatrix(F_direct_te.astype(np.float32))

        def to_direct_df(model):
            p = model.predict(dtest)
            if p.ndim == 1:
                p = p.reshape(-1, len(self._target_cols))
            return pd.DataFrame(p, index=X.index, columns=self._target_cols)[leg_raw.columns]

        def to_leg_target(model):
            p = model.predict(dtest)
            if p.ndim == 1:
                p = p.reshape(-1, len(self._leg_cols))
            return leg_to_target(p)

        xgb_direct_raw_df = to_direct_df(self._xgb_direct_raw)
        xgb_direct_dm_df = to_direct_df(self._xgb_direct_dm)
        xgb_leg_raw_target = to_leg_target(self._xgb_leg_raw)
        xgb_leg_dm_target = to_leg_target(self._xgb_leg_dm)

        return (
            _zscore(leg_raw) + _zscore(leg_dm)
            + _zscore(direct_raw) + _zscore(direct_dm)
            + _zscore(xgb_direct_raw_df) + _zscore(xgb_direct_dm_df)
            + _zscore(xgb_leg_raw_target) + _zscore(xgb_leg_dm_target)
        ) / 8.0
