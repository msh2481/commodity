"""v10: three-member ensemble (leg-ridge + direct-ridge + PLS on leg).

Adds a Partial Least Squares leg model as a third diverse member. PLS
finds latent components that maximize covariance with the leg returns
and behaves differently from ridge on wide feature matrices.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from features import wide_features
from utils import DATA_DIR, LAGS, compose_targets, leg_returns, load_target_pairs, unique_legs


ALPHA_LEG = 30.0
ALPHA_DIRECT = 10.0
PLS_N_COMPONENTS = 30


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

        # PLS on leg returns
        self._pls_leg = PLSRegression(n_components=PLS_N_COMPONENTS, scale=False)
        # PLSRegression expects float64 for numerical stability
        self._pls_leg.fit(F_tr.astype(np.float64), Y_leg_np.astype(np.float64))

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        F_te = self._F.loc[X.index].to_numpy(dtype=np.float32)
        F_te = np.nan_to_num(F_te, nan=0.0, posinf=0.0, neginf=0.0)
        F_te = self._scaler.transform(F_te)

        leg_pred = self._ridge_leg.predict(F_te)
        direct_pred = self._ridge_direct.predict(F_te)
        pls_leg_pred = self._pls_leg.predict(F_te.astype(np.float64))

        def leg_to_target(pred):
            df = pd.DataFrame(
                pred,
                index=X.index,
                columns=pd.MultiIndex.from_tuples(self._leg_cols, names=["asset", "lag"]),
            )
            return compose_targets(df, self.tp)

        leg_target = leg_to_target(leg_pred)
        pls_target = leg_to_target(pls_leg_pred)
        direct_df = pd.DataFrame(direct_pred, index=X.index, columns=self._target_cols)
        direct_df = direct_df[leg_target.columns]

        def zscore(df: pd.DataFrame) -> pd.DataFrame:
            mu = df.mean(axis=0)
            sd = df.std(axis=0).replace(0, 1.0)
            return (df - mu) / sd

        return (zscore(leg_target) + zscore(direct_df) + zscore(pls_target)) / 3.0
