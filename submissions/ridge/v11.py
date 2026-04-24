"""v11: feature bagging ensemble on top of v6.

Fit N_BAGS per-target ridges, each using a random ~75% subset of features, with
rank-y training. Average predictions (scale-matched since rank-y is in [-0.5, 0.5]).
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


ALPHA = 0.3
N_BAGS = 10
FEATURE_FRAC = 0.75
SEED = 42


class Model:
    device = "cpu"

    def __init__(self) -> None:
        self.alpha = ALPHA
        self._target_cols: list[str] | None = None
        self._per_target: dict[str, list[dict]] = {}

    @staticmethod
    def _xs_rank_targets(y: pd.DataFrame) -> pd.DataFrame:
        return y.rank(axis=1, pct=True) - 0.5

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        _prepare()
        self._target_cols = list(y.columns)
        y_rank = self._xs_rank_targets(y)
        train_index = X.index
        rng = np.random.default_rng(SEED)

        for t in self._target_cols:
            feats_full = _build_target_features(t)
            feat_train = feats_full.loc[train_index]
            y_t = y_rank[t].loc[train_index]
            mask = y_t.notna()
            Xf = feat_train.loc[mask].fillna(0.0).values
            yf = y_t.loc[mask].values.astype(np.float64)
            n_feats = Xf.shape[1]
            k = max(5, int(round(n_feats * FEATURE_FRAC)))
            bags = []
            cols_list = list(feat_train.columns)
            for _ in range(N_BAGS):
                idx = rng.choice(n_feats, size=k, replace=False)
                Xb = Xf[:, idx]
                mu = Xb.mean(axis=0)
                sd = Xb.std(axis=0)
                sd = np.where(sd < 1e-12, 1.0, sd)
                Xs = (Xb - mu) / sd
                reg = Ridge(alpha=self.alpha, fit_intercept=True)
                reg.fit(Xs, yf)
                bags.append({
                    "reg": reg, "mu": mu, "sd": sd, "idx": idx.astype(np.int32),
                })
            self._per_target[t] = {"bags": bags, "cols": cols_list}

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self._target_cols is not None
        test_index = X.index
        out = np.zeros((len(test_index), len(self._target_cols)), dtype=np.float64)
        for j, t in enumerate(self._target_cols):
            spec = self._per_target[t]
            feats = _build_target_features(t).loc[test_index].reindex(columns=spec["cols"])
            Xf = feats.fillna(0.0).values
            acc = np.zeros(len(test_index), dtype=np.float64)
            for bag in spec["bags"]:
                Xb = Xf[:, bag["idx"]]
                Xs = (Xb - bag["mu"]) / bag["sd"]
                Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
                acc += bag["reg"].predict(Xs)
            out[:, j] = acc / len(spec["bags"])
        return pd.DataFrame(out, index=test_index, columns=self._target_cols)
