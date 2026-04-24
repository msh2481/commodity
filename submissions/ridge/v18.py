"""v18: v14 + LASSO-based feature selection.

Replace correlation-based selection with LASSO (small alpha). Features with
non-zero LASSO coefs are kept; then fit ridge on those. Joint selection
accounts for feature correlations that univariate correlation misses.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso

from v4 import _build_target_features, _prepare


ALPHA_RIDGE = 0.3
ALPHA_LASSO = 0.001
MIN_KEEP = 8
MAX_KEEP = 50


class Model:
    device = "cpu"

    def __init__(self) -> None:
        self.alpha = ALPHA_RIDGE
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

            mu0 = Xf.mean(axis=0)
            sd0 = Xf.std(axis=0)
            sd0 = np.where(sd0 < 1e-12, 1.0, sd0)
            Xs0 = (Xf - mu0) / sd0

            lasso = Lasso(alpha=ALPHA_LASSO, max_iter=3000)
            lasso.fit(Xs0, yf)
            nonzero = np.where(np.abs(lasso.coef_) > 1e-12)[0]
            if len(nonzero) < MIN_KEEP:
                # Fall back to top-K by |corr|
                yf_c = yf - yf.mean()
                X_c = Xs0  # already standardized
                corr = (X_c * yf_c[:, None]).mean(axis=0) / (yf_c.std() + 1e-12)
                sel = np.argsort(-np.abs(corr))[:MIN_KEEP]
                sel = np.sort(sel)
            elif len(nonzero) > MAX_KEEP:
                # Keep top-MAX_KEEP by |lasso coef|
                order = np.argsort(-np.abs(lasso.coef_[nonzero]))[:MAX_KEEP]
                sel = np.sort(nonzero[order])
            else:
                sel = np.sort(nonzero)

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
