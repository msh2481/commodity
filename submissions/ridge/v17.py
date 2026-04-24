"""v17: v14 + row bootstrap bagging.

For each target, fit N_BAGS ridge models. Each uses a bootstrap sample of training
rows (with replacement), its own top-K feature selection by |corr|, and the same
rank-y target. Average predictions.
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
TOP_K = 30
N_BAGS = 8
SEED = 42


class Model:
    device = "cpu"

    def __init__(self) -> None:
        self.alpha = ALPHA
        self._target_cols: list[str] | None = None
        self._per_target: dict[str, dict] = {}

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
            Xf_all = feat_train.loc[mask].fillna(0.0).values.astype(np.float64)
            yf_all = y_t.loc[mask].values.astype(np.float64)
            cols = list(feat_train.columns)
            n = Xf_all.shape[0]
            n_feats = Xf_all.shape[1]
            k = min(TOP_K, n_feats)
            bags = []
            for _ in range(N_BAGS):
                idx = rng.integers(0, n, size=n)
                Xb = Xf_all[idx]
                yb = yf_all[idx]
                yb_c = yb - yb.mean()
                y_std = yb_c.std()
                if y_std < 1e-12:
                    sel = np.arange(n_feats)
                else:
                    X_c = Xb - Xb.mean(axis=0, keepdims=True)
                    x_std = X_c.std(axis=0)
                    x_std = np.where(x_std < 1e-12, 1.0, x_std)
                    corr = (X_c * yb_c[:, None]).mean(axis=0) / (x_std * y_std)
                    sel = np.argsort(-np.abs(corr))[:k]
                sel = np.sort(sel)
                Xb_sel = Xb[:, sel]
                mu = Xb_sel.mean(axis=0)
                sd = Xb_sel.std(axis=0)
                sd = np.where(sd < 1e-12, 1.0, sd)
                Xs = (Xb_sel - mu) / sd
                reg = Ridge(alpha=self.alpha, fit_intercept=True)
                reg.fit(Xs, yb)
                bags.append({
                    "reg": reg, "mu": mu, "sd": sd, "sel": sel.astype(np.int32),
                })
            self._per_target[t] = {"bags": bags, "cols": cols}

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self._target_cols is not None
        test_index = X.index
        out = np.zeros((len(test_index), len(self._target_cols)), dtype=np.float64)
        for j, t in enumerate(self._target_cols):
            spec = self._per_target[t]
            feats = _build_target_features(t).loc[test_index].reindex(columns=spec["cols"])
            Xf = feats.fillna(0.0).values.astype(np.float64)
            acc = np.zeros(len(test_index), dtype=np.float64)
            for bag in spec["bags"]:
                Xb = Xf[:, bag["sel"]]
                Xs = (Xb - bag["mu"]) / bag["sd"]
                Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
                acc += bag["reg"].predict(Xs)
            out[:, j] = acc / len(spec["bags"])
        return pd.DataFrame(out, index=test_index, columns=self._target_cols)
