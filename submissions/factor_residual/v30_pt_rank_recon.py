"""v30 — Per-target ridge with cross-sectional rank of the RECONSTRUCTED
label (sum of next ℓ V values), not the true y. This isolates the
rank-transform effect from the NaN/y-filtering effect (v14/v27 used true y
and suffered from it).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from v3 import _load_target_pairs, _target_assets, _log_returns, LAGS
from v28_pt_rank_xy import _xs_rank


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

    def _stack_lagged(self, V):
        T, N = V.shape
        out = np.zeros((T, N, self.L))
        for t in range(T):
            start = max(0, t - self.L + 1)
            block = V[start : t + 1]
            pad = self.L - block.shape[0]
            if pad > 0:
                block = np.vstack([np.zeros((pad, N)), block])
            out[t] = block.T
        return out

    def fit(self, X, y):
        prices = self._asset_prices(X).sort_index()
        R = _log_returns(prices).values
        V = self._target_values(R)
        feats = self._stack_lagged(V)
        T, N, D = feats.shape

        # Reconstruct ℓ-step label from V for each target column.
        # Build (T, N) matrix of reconstructed labels, NaN where out-of-range.
        labels = np.full((T, N), np.nan)
        for n in range(N):
            lag = int(self.pairs_.iloc[n]["lag"])
            if T - lag - 1 <= 0:
                continue
            tgt = np.zeros(T - lag)
            for i in range(1, lag + 1):
                tgt += V[i : T - lag + i, n]
            labels[: T - lag, n] = tgt

        # Cross-sectional rank across targets per date; standardized to ~[-1,1].
        y_rank = _xs_rank(labels)  # (T, N), NaN preserved

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
        feats = self._stack_lagged(V)
        w = self.coefs_[:, :-1]
        b = self.coefs_[:, -1]
        preds = np.einsum('tnd,nd->tn', feats, w) + b[None, :]
        return pd.DataFrame(preds, index=prices.index, columns=self.target_cols_)
