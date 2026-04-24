"""v20: direct per-target ridge alone (wide features).

Part of the hypothesis check: if the leg-composition structural prior
isn't helping, is direct ridge alone (fitted on target labels) the
strongest single model? Tune ALPHA_DIRECT.
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
from utils import DATA_DIR, load_target_pairs, unique_legs


ALPHA = 1000.0


class Model:
    device = "cpu"

    def __init__(self) -> None:
        self.tp = load_target_pairs()
        self.legs = unique_legs(self.tp)
        self.prices = (
            pd.read_csv(DATA_DIR / "train.csv").set_index("date_id").sort_index()
        )

    def fit(self, X, y):
        F = wide_features(self.prices, self.legs)
        self._F = F
        F_tr = F.loc[X.index].to_numpy(dtype=np.float32)
        F_tr = np.nan_to_num(F_tr, nan=0.0, posinf=0.0, neginf=0.0)
        self._scaler = StandardScaler()
        F_tr = self._scaler.fit_transform(F_tr)

        self._target_cols = list(y.columns)
        Y = np.nan_to_num(y.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        self._ridge = Ridge(alpha=ALPHA, fit_intercept=True)
        self._ridge.fit(F_tr, Y)

    def predict(self, X):
        F_te = self._F.loc[X.index].to_numpy(dtype=np.float32)
        F_te = np.nan_to_num(F_te, nan=0.0, posinf=0.0, neginf=0.0)
        F_te = self._scaler.transform(F_te)
        preds = self._ridge.predict(F_te)
        return pd.DataFrame(preds, index=X.index, columns=self._target_cols)
