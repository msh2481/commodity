"""v28 — Per-target ridge with rank-transformed features AND rank-transformed
target. Cross-sectional ranks on each day align both sides with the Spearman
metric and are outlier-robust."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from v3 import _load_target_pairs, _target_assets, _log_returns, LAGS


def _xs_rank(M: np.ndarray) -> np.ndarray:
    """Cross-sectional rank along axis=1 (across columns per row).
    Returns standardized ranks roughly in [-1, 1]. NaN stays NaN."""
    df = pd.DataFrame(M)
    r = df.rank(axis=1, method="average", na_option="keep")
    n_valid = r.notna().sum(axis=1).replace(0, 1)
    centered = r.sub((n_valid + 1) / 2, axis=0)
    normed = centered.div(n_valid / 2, axis=0)
    return normed.values


class Model:
    device = "cpu"
    L = 5
    alpha = 1.0

    def __init__(self):
        self.pairs_ = _load_target_pairs()
        self.assets_ = _target_assets(self.pairs_)
        self.target_cols_ = [f"target_{i}" for i in range(len(self.pairs_))]
        self.coefs_: np.ndarray | None = None

    def _asset_prices(self, X):
        cols = [a for a in self.assets_ if a in X.columns]
        return X[cols].copy()

    def _target_values(self, R):
        asset_to_idx = {a: i for i, a in enumerate(self.assets_)}
        T = R.shape[0]
        vals = np.zeros((T, len(self.pairs_)))
        for idx, row in self.pairs_.iterrows():
            pair = row["pair"]
            if " - " in pair:
                a, b = [s.strip() for s in pair.split(" - ")]
                vals[:, idx] = R[:, asset_to_idx[a]] - R[:, asset_to_idx[b]]
            else:
                vals[:, idx] = R[:, asset_to_idx[pair.strip()]]
        return vals

    def _build_feats(self, V: np.ndarray) -> np.ndarray:
        """Per (t, target) features: lagged V values, each first rank-transformed
        cross-sectionally across targets."""
        T, N = V.shape
        feats = np.zeros((T, N, self.L))
        # precompute lagged V matrices first, then rank each cross-sectionally.
        for l in range(self.L):
            shifted = np.zeros((T, N))
            shifted[l:] = V[:T - l]
            # cross-sectional rank: ranks relative to other targets on same date
            feats[:, :, l] = np.nan_to_num(_xs_rank(shifted), nan=0.0)
        return feats

    def fit(self, X, y):
        prices = self._asset_prices(X).sort_index()
        R = _log_returns(prices).values
        V = self._target_values(R)
        feats = self._build_feats(V)
        T, N, D = feats.shape

        y_aligned = y.reindex(prices.index)[self.target_cols_].values
        y_rank = _xs_rank(y_aligned)  # (T, N)

        coefs = np.zeros((N, D + 1))
        for n in range(N):
            mask = ~np.isnan(y_rank[:, n])
            if mask.sum() < self.L + 1:
                continue
            Xn = feats[mask, n, :]
            yn = y_rank[mask, n]
            r = Ridge(alpha=self.alpha)
            r.fit(Xn, yn)
            coefs[n, :D] = r.coef_
            coefs[n, -1] = r.intercept_
        self.coefs_ = coefs

    def predict(self, X):
        prices = self._asset_prices(X).sort_index()
        R = _log_returns(prices).values
        V = self._target_values(R)
        feats = self._build_feats(V)
        w = self.coefs_[:, :-1]
        b = self.coefs_[:, -1]
        preds = np.einsum('tnd,nd->tn', feats, w) + b[None, :]
        return pd.DataFrame(preds, index=prices.index, columns=self.target_cols_)
