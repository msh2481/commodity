"""v21: ensemble of v14 at multiple K values.

Average per-date ranks of v14 K=30 and v14 K=45 predictions. Each K captures
slightly different feature set; ensemble may reduce variance.
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
K_VALUES = [30, 45]


class _V14Slot:
    def __init__(self, alpha: float, top_k: int) -> None:
        self.alpha = alpha
        self.top_k = top_k
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
        for t in self._target_cols:
            feats_full = _build_target_features(t)
            feat_train = feats_full.loc[train_index]
            y_t = y_rank[t].loc[train_index]
            mask = y_t.notna()
            Xf = feat_train.loc[mask].fillna(0.0).values.astype(np.float64)
            yf = y_t.loc[mask].values.astype(np.float64)
            cols = list(feat_train.columns)
            n_feats = Xf.shape[1]
            k = min(self.top_k, n_feats)
            yf_c = yf - yf.mean()
            y_std = yf_c.std()
            if y_std < 1e-12:
                sel = np.arange(n_feats)
            else:
                X_c = Xf - Xf.mean(axis=0, keepdims=True)
                x_std = X_c.std(axis=0)
                x_std = np.where(x_std < 1e-12, 1.0, x_std)
                corr = (X_c * yf_c[:, None]).mean(axis=0) / (x_std * y_std)
                sel = np.argsort(-np.abs(corr))[:k]
            sel = np.sort(sel)
            Xf_sel = Xf[:, sel]
            mu = Xf_sel.mean(axis=0)
            sd = Xf_sel.std(axis=0)
            sd = np.where(sd < 1e-12, 1.0, sd)
            Xs = (Xf_sel - mu) / sd
            reg = Ridge(alpha=self.alpha, fit_intercept=True)
            reg.fit(Xs, yf)
            self._per_target[t] = {
                "reg": reg, "mu": mu, "sd": sd, "cols": cols, "sel": sel.astype(np.int32),
            }

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self._target_cols is not None
        test_index = X.index
        out = np.zeros((len(test_index), len(self._target_cols)), dtype=np.float64)
        for j, t in enumerate(self._target_cols):
            spec = self._per_target[t]
            feats = _build_target_features(t).loc[test_index].reindex(columns=spec["cols"])
            Xf = feats.fillna(0.0).values.astype(np.float64)
            Xf = Xf[:, spec["sel"]]
            Xs = (Xf - spec["mu"]) / spec["sd"]
            Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
            out[:, j] = spec["reg"].predict(Xs)
        return pd.DataFrame(out, index=test_index, columns=self._target_cols)


class Model:
    device = "cpu"

    def __init__(self) -> None:
        self._slots = [_V14Slot(ALPHA, k) for k in K_VALUES]

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        for s in self._slots:
            s.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        preds = [s.predict(X) for s in self._slots]
        ranks = [(p.rank(axis=1, pct=True) - 0.5) for p in preds]
        return sum(ranks) / len(ranks)
