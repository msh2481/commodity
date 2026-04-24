"""v15 — Per-target ridge with multi-scale momentum features.

Features per target: lagged 1-day values + 5-day sum + 20-day sum + 60-day sum.
Captures both short-term mean-reversion and longer-term trend."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from v3 import _load_target_pairs, _target_assets, _log_returns, LAGS


class Model:
    device = "cpu"
    L = 5            # lagged 1-day values
    windows = (5, 10, 20, 60)  # rolling cumulative
    alpha = 1.0

    def __init__(self):
        self.pairs_ = _load_target_pairs()
        self.assets_ = _target_assets(self.pairs_)
        self.target_cols_ = [f"target_{i}" for i in range(len(self.pairs_))]
        self.coefs_: np.ndarray | None = None

    def _asset_prices(self, X):
        cols = [a for a in self.assets_ if a in X.columns]
        return X[cols].copy()

    def _target_values(self, R: np.ndarray) -> np.ndarray:
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

    def _rolling_sum(self, V: np.ndarray, w: int) -> np.ndarray:
        """Past-w-day sum (exclusive of today). Row t uses V[max(0,t-w):t]."""
        T, N = V.shape
        cs = np.concatenate([np.zeros((1, N)), np.cumsum(V, axis=0)], axis=0)
        # cs[i] = sum V[:i]
        out = np.zeros((T, N))
        for t in range(T):
            start = max(0, t - w)
            out[t] = cs[t] - cs[start]
        return out

    def _build_feats(self, V: np.ndarray) -> np.ndarray:
        """Out: (T, N, D). Each row has L+len(windows) features per target.
        Features: V[t-l] for l=0..L-1, then rolling_sum_w for each w."""
        T, N = V.shape
        D = self.L + len(self.windows)
        feats = np.zeros((T, N, D))
        # Lagged
        for l in range(self.L):
            shifted = np.zeros((T, N))
            shifted[l:] = V[:T - l]
            feats[:, :, l] = shifted
        # Rolling sums (lagged by 1 to avoid same-day leakage for predict;
        # but V itself at t uses prices at t-1,t - so rolling_sum through t
        # is non-leaky too. Use t-exclusive to strictly use past info.)
        for i, w in enumerate(self.windows):
            feats[:, :, self.L + i] = self._rolling_sum(V, w)
        return feats

    def fit(self, X, y):
        prices = self._asset_prices(X).sort_index()
        R = _log_returns(prices).values
        V = self._target_values(R)
        feats = self._build_feats(V)  # (T, N, D)
        T, N, D = feats.shape

        # target = sum of next lag V values; use reconstructed (v10-style) since
        # it works better empirically than true y here.
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
