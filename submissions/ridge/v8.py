"""v8: ensemble of rank-y (v6) and raw-y (v4) per-target ridges.

Fit both. At predict time, convert each model's outputs to per-date ranks,
then average. Average-of-ranks is the rank-invariant ensemble.
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

from v4 import _build_target_features, _prepare


ALPHA_RANK = 1.0
ALPHA_RAW = 10.0


def _fit_per_target(
    y_mat: pd.DataFrame,
    X: pd.DataFrame,
    alpha: float,
) -> dict:
    per_target = {}
    train_index = X.index
    for t in y_mat.columns:
        feats_full = _build_target_features(t)
        feat_train = feats_full.loc[train_index]
        y_t = y_mat[t].loc[train_index]
        mask = y_t.notna()
        Xf = feat_train.loc[mask].fillna(0.0).values
        yf = y_t.loc[mask].values.astype(np.float64)
        mu = Xf.mean(axis=0)
        sd = Xf.std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        Xs = (Xf - mu) / sd
        reg = Ridge(alpha=alpha, fit_intercept=True)
        reg.fit(Xs, yf)
        per_target[t] = {
            "reg": reg, "mu": mu, "sd": sd, "cols": list(feat_train.columns),
        }
    return per_target


def _predict_per_target(per_target: dict, X: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
    test_index = X.index
    out = np.zeros((len(test_index), len(target_cols)), dtype=np.float64)
    for j, t in enumerate(target_cols):
        spec = per_target[t]
        feats = _build_target_features(t).loc[test_index].reindex(columns=spec["cols"])
        Xf = feats.fillna(0.0).values
        Xs = (Xf - spec["mu"]) / spec["sd"]
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        out[:, j] = spec["reg"].predict(Xs)
    return pd.DataFrame(out, index=test_index, columns=target_cols)


class Model:
    device = "cpu"

    def __init__(self) -> None:
        self._target_cols: list[str] | None = None
        self._models: dict[str, dict] = {}

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        _prepare()
        self._target_cols = list(y.columns)
        y_rank = y.rank(axis=1, pct=True) - 0.5
        self._models["rank"] = _fit_per_target(y_rank, X, ALPHA_RANK)
        self._models["raw"] = _fit_per_target(y, X, ALPHA_RAW)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self._target_cols is not None
        p_rank = _predict_per_target(self._models["rank"], X, self._target_cols)
        p_raw = _predict_per_target(self._models["raw"], X, self._target_cols)
        # Rank each per-date then average
        r_rank = p_rank.rank(axis=1, pct=True) - 0.5
        r_raw = p_raw.rank(axis=1, pct=True) - 0.5
        ensemble = 0.5 * (r_rank + r_raw)
        return ensemble
