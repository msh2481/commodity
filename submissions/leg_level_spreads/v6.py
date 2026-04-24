"""v6: XGBoost GPU leg model.

Non-linear leg model using XGBoost with multi_strategy='multi_output_tree'
so all 424 (asset, lag) outputs train from a single booster per boosting
round. GPU device. Same wide feature set as v5.
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
from sklearn.preprocessing import StandardScaler

from features import wide_features
from utils import DATA_DIR, LAGS, compose_targets, leg_returns, load_target_pairs, unique_legs


class Model:
    device = "cuda"

    def __init__(self) -> None:
        self.tp = load_target_pairs()
        self.legs = unique_legs(self.tp)
        self.prices = (
            pd.read_csv(DATA_DIR / "train.csv").set_index("date_id").sort_index()
        )
        self._scaler: StandardScaler | None = None
        self._booster: xgb.Booster | None = None
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
        F_tr = self._scaler.fit_transform(F_tr).astype(np.float32)

        dtrain = xgb.DMatrix(F_tr, label=Y_tr)
        params = {
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
        self._booster = xgb.train(params, dtrain, num_boost_round=100)
        self._F = F

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self._scaler is not None and self._booster is not None
        assert self._leg_cols is not None
        F_te = self._F.loc[X.index].to_numpy(dtype=np.float32)
        F_te = np.nan_to_num(F_te, nan=0.0, posinf=0.0, neginf=0.0)
        F_te = self._scaler.transform(F_te).astype(np.float32)

        dtest = xgb.DMatrix(F_te)
        preds = self._booster.predict(dtest)
        if preds.ndim == 1:
            preds = preds.reshape(-1, len(self._leg_cols))

        leg_preds = pd.DataFrame(
            preds,
            index=X.index,
            columns=pd.MultiIndex.from_tuples(self._leg_cols, names=["asset", "lag"]),
        )
        return compose_targets(leg_preds, self.tp)
