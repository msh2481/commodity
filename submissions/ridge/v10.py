"""v10: pooled long-format ridge.

Stack (date, target) training rows into one huge matrix. Features are pair-specific
(A, B, spread) plus lag one-hot plus market context. One single ridge is fit across
all 420 spread targets × ~1470 dates. Single-asset targets (4) predicted as 0.

Hypothesis: pooling vastly increases training size; generalization improves at the
cost of losing some per-pair specificity. Features encode pair identity via their
values, not a per-pair indicator.
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


ALPHA = 10.0


def _xs_rank_targets(y: pd.DataFrame) -> pd.DataFrame:
    return y.rank(axis=1, pct=True) - 0.5


def _lag_onehot(lag: int) -> dict[str, float]:
    return {f"lag{L}": 1.0 if L == lag else 0.0 for L in (1, 2, 3, 4)}


class Model:
    device = "cpu"

    def __init__(self) -> None:
        self.alpha = ALPHA
        self._target_cols: list[str] | None = None
        self._reg: Ridge | None = None
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None
        self._feat_cols: list[str] | None = None
        self._pair_info: pd.DataFrame | None = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        _prepare()
        self._target_cols = list(y.columns)
        pairs = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "target_pairs.csv")
        pairs = pairs.set_index("target")
        self._pair_info = pairs

        y_rank = _xs_rank_targets(y)
        train_index = X.index

        all_feat_cols: list[str] | None = None
        Xs_parts: list[np.ndarray] = []
        ys_parts: list[np.ndarray] = []

        for t in self._target_cols:
            if " - " not in pairs.loc[t, "pair"]:
                continue
            feats_full = _build_target_features(t)
            feat_train = feats_full.loc[train_index].copy()
            y_t = y_rank[t].loc[train_index]
            mask = y_t.notna()
            if mask.sum() < 20:
                continue
            lag = int(pairs.loc[t, "lag"])
            one = _lag_onehot(lag)
            for k, v in one.items():
                feat_train[k] = v
            if all_feat_cols is None:
                all_feat_cols = list(feat_train.columns)
            feat_train = feat_train.reindex(columns=all_feat_cols).fillna(0.0)
            Xs_parts.append(feat_train.loc[mask].values.astype(np.float64))
            ys_parts.append(y_t.loc[mask].values.astype(np.float64))

        Xf = np.concatenate(Xs_parts, axis=0)
        yf = np.concatenate(ys_parts, axis=0)
        self._feat_cols = all_feat_cols
        self._mu = Xf.mean(axis=0)
        self._sd = Xf.std(axis=0)
        self._sd = np.where(self._sd < 1e-12, 1.0, self._sd)
        Xs = (Xf - self._mu) / self._sd
        self._reg = Ridge(alpha=self.alpha, fit_intercept=True)
        self._reg.fit(Xs, yf)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self._target_cols is not None and self._reg is not None
        test_index = X.index
        pairs = self._pair_info
        out = np.zeros((len(test_index), len(self._target_cols)), dtype=np.float64)
        for j, t in enumerate(self._target_cols):
            if " - " not in pairs.loc[t, "pair"]:
                continue
            feats = _build_target_features(t).loc[test_index].copy()
            lag = int(pairs.loc[t, "lag"])
            one = _lag_onehot(lag)
            for k, v in one.items():
                feats[k] = v
            feats = feats.reindex(columns=self._feat_cols).fillna(0.0)
            Xs = (feats.values.astype(np.float64) - self._mu) / self._sd
            Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
            out[:, j] = self._reg.predict(Xs)
        return pd.DataFrame(out, index=test_index, columns=self._target_cols)
