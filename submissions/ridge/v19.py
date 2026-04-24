"""v19: principal component regression per target.

Per-target: PCA on training features, keep top-K components, fit ridge on those
plus the top-M features selected by |corr|. Compact features but preserves
combined signal.
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
from sklearn.decomposition import PCA

from v4 import _build_target_features, _prepare


ALPHA = 0.3
N_PC = 15
TOP_K_RAW = 15


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
        for t in self._target_cols:
            feats_full = _build_target_features(t)
            feat_train = feats_full.loc[train_index]
            y_t = y_rank[t].loc[train_index]
            mask = y_t.notna()
            Xf = feat_train.loc[mask].fillna(0.0).values.astype(np.float64)
            yf = y_t.loc[mask].values.astype(np.float64)
            cols = list(feat_train.columns)
            n_feats = Xf.shape[1]

            # Standardize full feature matrix
            mu0 = Xf.mean(axis=0)
            sd0 = Xf.std(axis=0)
            sd0 = np.where(sd0 < 1e-12, 1.0, sd0)
            Xs0 = (Xf - mu0) / sd0

            # PCA top-N_PC components
            n_pc = min(N_PC, n_feats, Xs0.shape[0] - 1)
            pca = PCA(n_components=n_pc)
            Xs_pca = pca.fit_transform(Xs0)

            # Top-K raw features by |corr|
            k_raw = min(TOP_K_RAW, n_feats)
            yf_c = yf - yf.mean()
            y_std = yf_c.std()
            if y_std > 1e-12:
                corr = (Xs0 * yf_c[:, None]).mean(axis=0) / (y_std + 1e-12)
                sel = np.argsort(-np.abs(corr))[:k_raw]
                sel = np.sort(sel)
            else:
                sel = np.arange(k_raw)
            Xs_raw = Xs0[:, sel]

            # Combine PC + selected raw
            Xs_combined = np.concatenate([Xs_pca, Xs_raw], axis=1)

            reg = Ridge(alpha=self.alpha, fit_intercept=True)
            reg.fit(Xs_combined, yf)
            self._per_target[t] = {
                "reg": reg,
                "mu0": mu0, "sd0": sd0,
                "pca": pca,
                "sel": sel.astype(np.int32),
                "cols": cols,
            }

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self._target_cols is not None
        test_index = X.index
        out = np.zeros((len(test_index), len(self._target_cols)), dtype=np.float64)
        for j, t in enumerate(self._target_cols):
            spec = self._per_target[t]
            feats = _build_target_features(t).loc[test_index].reindex(columns=spec["cols"])
            Xf = feats.fillna(0.0).values.astype(np.float64)
            Xs0 = (Xf - spec["mu0"]) / spec["sd0"]
            Xs0 = np.nan_to_num(Xs0, nan=0.0, posinf=0.0, neginf=0.0)
            Xs_pca = spec["pca"].transform(Xs0)
            Xs_raw = Xs0[:, spec["sel"]]
            Xs_combined = np.concatenate([Xs_pca, Xs_raw], axis=1)
            out[:, j] = spec["reg"].predict(Xs_combined)
        return pd.DataFrame(out, index=test_index, columns=self._target_cols)
