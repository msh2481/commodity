"""v15: three-way ensemble — ridge_leg + ridge_direct + xgboost_leg.

XGBoost contributes non-linear leg predictions that decorrelate from
ridge. All three members are z-scored per target column, then averaged
with weights (WEIGHT_LEG, WEIGHT_DIRECT, WEIGHT_XGB).
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

from features import wide_features
from utils import DATA_DIR, LAGS, compose_targets, leg_returns, load_target_pairs, unique_legs


ALPHA_LEG = 30.0
ALPHA_DIRECT = 10.0

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "device": "cuda",
    "multi_strategy": "one_output_per_tree",
    "max_depth": 4,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "reg_lambda": 10.0,
    "verbosity": 0,
}
XGB_ROUNDS = 80

WEIGHT_LEG = 1.0
WEIGHT_DIRECT = 1.0
WEIGHT_XGB = 1.0


class Model:
    device = "cuda"

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
        F_tr = self._scaler.fit_transform(F_tr).astype(np.float32)

        Y_leg = leg_returns(self.prices, self.legs, lags=LAGS).loc[X.index]
        self._leg_cols = list(Y_leg.columns)
        Y_leg_np = np.nan_to_num(Y_leg.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        self._ridge_leg = Ridge(alpha=ALPHA_LEG, fit_intercept=True)
        self._ridge_leg.fit(F_tr, Y_leg_np)

        self._target_cols = list(y.columns)
        Y_direct = np.nan_to_num(y.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        self._ridge_direct = Ridge(alpha=ALPHA_DIRECT, fit_intercept=True)
        self._ridge_direct.fit(F_tr, Y_direct)

        dtrain = xgb.DMatrix(F_tr, label=Y_leg_np)
        self._xgb = xgb.train(XGB_PARAMS, dtrain, num_boost_round=XGB_ROUNDS)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        F_te = self._F.loc[X.index].to_numpy(dtype=np.float32)
        F_te = np.nan_to_num(F_te, nan=0.0, posinf=0.0, neginf=0.0)
        F_te = self._scaler.transform(F_te).astype(np.float32)

        leg_pred = self._ridge_leg.predict(F_te)
        direct_pred = self._ridge_direct.predict(F_te)

        dtest = xgb.DMatrix(F_te)
        xgb_pred = self._xgb.predict(dtest)
        if xgb_pred.ndim == 1:
            xgb_pred = xgb_pred.reshape(-1, len(self._leg_cols))

        def to_leg_df(arr):
            return pd.DataFrame(
                arr,
                index=X.index,
                columns=pd.MultiIndex.from_tuples(self._leg_cols, names=["asset", "lag"]),
            )

        leg_target = compose_targets(to_leg_df(leg_pred), self.tp)
        xgb_target = compose_targets(to_leg_df(xgb_pred), self.tp)
        direct_df = pd.DataFrame(direct_pred, index=X.index, columns=self._target_cols)
        direct_df = direct_df[leg_target.columns]

        def zscore(df: pd.DataFrame) -> pd.DataFrame:
            mu = df.mean(axis=0)
            sd = df.std(axis=0).replace(0, 1.0)
            return (df - mu) / sd

        total = WEIGHT_LEG + WEIGHT_DIRECT + WEIGHT_XGB
        combined = (
            WEIGHT_LEG * zscore(leg_target)
            + WEIGHT_DIRECT * zscore(direct_df)
            + WEIGHT_XGB * zscore(xgb_target)
        ) / total
        return combined
