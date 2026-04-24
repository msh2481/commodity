"""v2: ridge leg model, minimal features.

Multi-output ridge over all 106 × 4 = 424 (asset, lag) leg returns with
a minimal feature set (lagged log-returns at 1, 5, 20 days for each leg
asset). Compose to target_0..target_423 via target_pairs.csv.
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

from features import minimal_features
from utils import DATA_DIR, LAGS, compose_targets, leg_returns, load_target_pairs, unique_legs


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

    def _build_feature_panel(self) -> pd.DataFrame:
        feats = minimal_features(self.prices, self.legs)
        return feats

    def _build_label_panel(self) -> pd.DataFrame:
        legs_df = leg_returns(self.prices, self.legs, lags=LAGS)
        return legs_df

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        F = self._build_feature_panel()
        Y = self._build_label_panel()

        train_idx = X.index
        F_tr = F.loc[train_idx].to_numpy(dtype=np.float32)
        Y_tr = Y.loc[train_idx]
        self._leg_cols = list(Y_tr.columns)
        Y_tr = Y_tr.to_numpy(dtype=np.float32)

        F_tr = np.nan_to_num(F_tr, nan=0.0, posinf=0.0, neginf=0.0)
        Y_tr = np.nan_to_num(Y_tr, nan=0.0, posinf=0.0, neginf=0.0)

        self._scaler = StandardScaler()
        F_tr = self._scaler.fit_transform(F_tr)

        self._ridge = Ridge(alpha=10.0, fit_intercept=True)
        self._ridge.fit(F_tr, Y_tr)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self._scaler is not None and self._ridge is not None
        assert self._leg_cols is not None
        F = self._build_feature_panel()
        F_te = F.loc[X.index].to_numpy(dtype=np.float32)
        F_te = np.nan_to_num(F_te, nan=0.0, posinf=0.0, neginf=0.0)
        F_te = self._scaler.transform(F_te)
        preds = self._ridge.predict(F_te)

        leg_preds = pd.DataFrame(
            preds, index=X.index, columns=pd.MultiIndex.from_tuples(self._leg_cols, names=["asset", "lag"])
        )
        return compose_targets(leg_preds, self.tp)
