"""v28: bigger ensemble - K={30,45} without weights + multiple v26 halflives."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pandas as pd
from v21 import _V14Slot

import numpy as np
from sklearn.linear_model import Ridge
from v4 import _build_target_features, _prepare


ALPHA = 0.3
TOP_K = 30


class _WeightedSlot:
    def __init__(self, halflife: float, top_k: int = TOP_K, alpha: float = ALPHA) -> None:
        self.halflife = halflife
        self.top_k = top_k
        self.alpha = alpha
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
        lam = np.log(2) / self.halflife
        for t in self._target_cols:
            feats_full = _build_target_features(t)
            feat_train = feats_full.loc[train_index]
            y_t = y_rank[t].loc[train_index]
            mask = y_t.notna()
            dates = feat_train.index.values[mask.values]
            max_d = dates.max()
            w = np.exp(-lam * (max_d - dates))
            Xf = feat_train.loc[mask].fillna(0.0).values.astype(np.float64)
            yf = y_t.loc[mask].values.astype(np.float64)
            cols = list(feat_train.columns)
            n_feats = Xf.shape[1]
            k = min(self.top_k, n_feats)
            w_sum = w.sum()
            mu_y = (w * yf).sum() / w_sum
            yf_c = yf - mu_y
            y_std = np.sqrt((w * yf_c ** 2).sum() / w_sum)
            if y_std < 1e-12:
                sel = np.arange(n_feats)
            else:
                mu_x = (w[:, None] * Xf).sum(axis=0) / w_sum
                X_c = Xf - mu_x
                x_std = np.sqrt((w[:, None] * X_c ** 2).sum(axis=0) / w_sum)
                x_std = np.where(x_std < 1e-12, 1.0, x_std)
                corr = (w[:, None] * X_c * yf_c[:, None]).sum(axis=0) / (w_sum * x_std * y_std)
                sel = np.argsort(-np.abs(corr))[:k]
            sel = np.sort(sel)
            Xf_sel = Xf[:, sel]
            mu = (w[:, None] * Xf_sel).sum(axis=0) / w_sum
            sd = np.sqrt((w[:, None] * (Xf_sel - mu) ** 2).sum(axis=0) / w_sum)
            sd = np.where(sd < 1e-12, 1.0, sd)
            Xs = (Xf_sel - mu) / sd
            reg = Ridge(alpha=self.alpha, fit_intercept=True)
            reg.fit(Xs, yf, sample_weight=w)
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
        self._slots = [
            _V14Slot(ALPHA, 30),
            _V14Slot(ALPHA, 45),
            _WeightedSlot(halflife=800),
            _WeightedSlot(halflife=1000),
            _WeightedSlot(halflife=1200),
        ]

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        for s in self._slots:
            s.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        preds = [s.predict(X) for s in self._slots]
        ranks = [(p.rank(axis=1, pct=True) - 0.5) for p in preds]
        return sum(ranks) / len(ranks)
