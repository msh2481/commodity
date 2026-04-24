"""v6: train on cross-sectionally rank-transformed y.

Same features as v4. At fit time, transform y[t, j] -> rank of target_j's
return among the 424 targets on date t, scaled to [-0.5, 0.5]. Train ridge
per-target to predict this rank. Since the metric is per-date Spearman across
targets, this loss is better aligned.
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

from v4 import ALPHA, _build_target_features, _prepare


class Model:
    device = "cpu"

    def __init__(self) -> None:
        self.alpha = ALPHA
        self._target_cols: list[str] | None = None
        self._per_target: dict[str, dict] = {}

    @staticmethod
    def _xs_rank_targets(y: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional rank per date, centered at 0, scale [-0.5, 0.5]."""
        ranks = y.rank(axis=1, pct=True) - 0.5
        return ranks

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        _prepare()
        self._target_cols = list(y.columns)
        y_rank = self._xs_rank_targets(y)
        train_index = X.index
        for t in self._target_cols:
            feats_full = _build_target_features(t)
            feat_train = feats_full.loc[train_index]
            y_t = y_rank[t].loc[train_index]
            mask = y_t.notna()
            Xf = feat_train.loc[mask].fillna(0.0).values
            yf = y_t.loc[mask].values.astype(np.float64)
            mu = Xf.mean(axis=0)
            sd = Xf.std(axis=0)
            sd = np.where(sd < 1e-12, 1.0, sd)
            Xs = (Xf - mu) / sd
            reg = Ridge(alpha=self.alpha, fit_intercept=True)
            reg.fit(Xs, yf)
            self._per_target[t] = {
                "reg": reg, "mu": mu, "sd": sd, "cols": list(feat_train.columns),
            }

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self._target_cols is not None
        test_index = X.index
        out = np.zeros((len(test_index), len(self._target_cols)), dtype=np.float64)
        for j, t in enumerate(self._target_cols):
            spec = self._per_target[t]
            feats = _build_target_features(t).loc[test_index].reindex(columns=spec["cols"])
            Xf = feats.fillna(0.0).values
            Xs = (Xf - spec["mu"]) / spec["sd"]
            Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
            out[:, j] = spec["reg"].predict(Xs)
        return pd.DataFrame(out, index=test_index, columns=self._target_cols)
