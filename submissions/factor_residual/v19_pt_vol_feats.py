"""v19 — Per-target ridge with cross-sectional rank + volatility features.

Adds to v10 features:
  - rolling_std(V, 20) as a volatility feature (size of recent moves)
  - cross-sectional rank of V[t] across targets on day t
Thesis: scale-aware mean-reversion + market-mood rank.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from v3 import _load_target_pairs, _target_assets, _log_returns, LAGS


class Model:
    device = "cpu"
    L = 5
    vol_w = 20
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

    def _rolling_std(self, V):
        T, N = V.shape
        out = np.ones((T, N)) * 1e-4
        for t in range(1, T):
            start = max(0, t - self.vol_w)
            block = V[start:t]
            if block.shape[0] >= 2:
                out[t] = np.maximum(block.std(axis=0), 1e-6)
        return out

    def _build_feats(self, V):
        """Out: (T, N, D) with D = L + 2.
        Features: lagged V[t-l] for l=0..L-1, rolling_std_20, cross-section rank of V[t]."""
        T, N = V.shape
        feats = np.zeros((T, N, self.L + 2))
        for l in range(self.L):
            shifted = np.zeros((T, N))
            shifted[l:] = V[:T - l]
            feats[:, :, l] = shifted
        feats[:, :, self.L] = self._rolling_std(V)
        # cross-sectional rank of V[t] across targets, centered and normalized
        ranks = pd.DataFrame(V).rank(axis=1, method="average").values
        ranks = (ranks - (N + 1) / 2) / (N / 2)  # roughly in [-1, 1]
        feats[:, :, self.L + 1] = ranks
        return feats

    def fit(self, X, y):
        prices = self._asset_prices(X).sort_index()
        R = _log_returns(prices).values
        V = self._target_values(R)
        feats = self._build_feats(V)
        T, N, D = feats.shape
        coefs = np.zeros((N, D + 1))
        for n in range(N):
            lag = int(self.pairs_.iloc[n]["lag"])
            if T - lag - 1 <= 0:
                continue
            tgt = np.zeros(T - lag)
            for i in range(1, lag + 1):
                tgt += V[i : T - lag + i, n]
            Xn = feats[: T - lag, n, :]
            r = Ridge(alpha=self.alpha)
            r.fit(Xn, tgt)
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
