"""v13: bagged ensemble ridge (leg + direct, feature-subsampled).

Trains N_BAGS ridge pairs, each on a random 60% feature subsample, then
averages z-scored predictions. Implements implicit bagging to reduce
variance of the ridge fits.
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
N_BAGS = 5
FEATURE_FRACTION = 0.6
BAG_SEED = 7


class Model:
    device = "cpu"
    cpus_per_fold = 64

    def __init__(self) -> None:
        self.tp = load_target_pairs()
        self.legs = unique_legs(self.tp)
        self.prices = (
            pd.read_csv(DATA_DIR / "train.csv").set_index("date_id").sort_index()
        )

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        F = wide_features(self.prices, self.legs)
        self._F = F
        F_tr_full = F.loc[X.index].to_numpy(dtype=np.float32)
        F_tr_full = np.nan_to_num(F_tr_full, nan=0.0, posinf=0.0, neginf=0.0)
        self._scaler = StandardScaler()
        F_tr_full = self._scaler.fit_transform(F_tr_full)

        Y_leg = leg_returns(self.prices, self.legs, lags=LAGS).loc[X.index]
        self._leg_cols = list(Y_leg.columns)
        Y_leg_np = np.nan_to_num(Y_leg.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        self._target_cols = list(y.columns)
        Y_direct = np.nan_to_num(y.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        rng = np.random.default_rng(BAG_SEED)
        n_feats = F_tr_full.shape[1]
        n_keep = int(n_feats * FEATURE_FRACTION)

        self._bags_leg: list[tuple[np.ndarray, Ridge]] = []
        self._bags_direct: list[tuple[np.ndarray, Ridge]] = []
        for b in range(N_BAGS):
            idx_leg = rng.choice(n_feats, size=n_keep, replace=False)
            r_leg = Ridge(alpha=ALPHA_LEG, fit_intercept=True)
            r_leg.fit(F_tr_full[:, idx_leg], Y_leg_np)
            self._bags_leg.append((idx_leg, r_leg))

            idx_dir = rng.choice(n_feats, size=n_keep, replace=False)
            r_dir = Ridge(alpha=ALPHA_DIRECT, fit_intercept=True)
            r_dir.fit(F_tr_full[:, idx_dir], Y_direct)
            self._bags_direct.append((idx_dir, r_dir))

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        F_te = self._F.loc[X.index].to_numpy(dtype=np.float32)
        F_te = np.nan_to_num(F_te, nan=0.0, posinf=0.0, neginf=0.0)
        F_te = self._scaler.transform(F_te)

        # average bagged leg predictions
        leg_sum = 0.0
        for idx, r in self._bags_leg:
            leg_sum = leg_sum + r.predict(F_te[:, idx])
        leg_pred = leg_sum / N_BAGS

        dir_sum = 0.0
        for idx, r in self._bags_direct:
            dir_sum = dir_sum + r.predict(F_te[:, idx])
        direct_pred = dir_sum / N_BAGS

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
