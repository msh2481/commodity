"""v14 — Per-target ridge using true y labels (not reconstructed).

Same feature design as v10 (lagged 1-day target values) but fit directly on
y[target_i], dropping NaN rows. This handles trading-holiday NaNs correctly
and uses true labels near the training-window edge."""

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
    alpha = 1.0

    def __init__(self) -> None:
        self.pairs_ = _load_target_pairs()
        self.assets_ = _target_assets(self.pairs_)
        self.target_cols_ = [f"target_{i}" for i in range(len(self.pairs_))]
        # per-target (n_targets, L+1)
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

    def _stack_lagged(self, V: np.ndarray) -> np.ndarray:
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

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        prices = self._asset_prices(X).sort_index()
        R = _log_returns(prices).values
        V = self._target_values(R)  # (T, N)
        feats = self._stack_lagged(V)  # (T, N, L)

        # Align y to X index
        y_aligned = y.reindex(prices.index)[self.target_cols_].values  # (T, N)
        N = V.shape[1]
        coefs = np.zeros((N, self.L + 1))
        for n in range(N):
            mask = ~np.isnan(y_aligned[:, n])
            if mask.sum() < self.L + 1:
                continue
            Xn = feats[mask, n, :]
            yn = y_aligned[mask, n]
            r = Ridge(alpha=self.alpha)
            r.fit(Xn, yn)
            coefs[n, :self.L] = r.coef_
            coefs[n, -1] = r.intercept_
        self.coefs_ = coefs

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        prices = self._asset_prices(X).sort_index()
        R = _log_returns(prices).values
        V = self._target_values(R)
        feats = self._stack_lagged(V)
        T_test, N, _ = feats.shape

        w = self.coefs_[:, :self.L]  # (N, L)
        b = self.coefs_[:, -1]  # (N,)
        preds = np.einsum('tnl,nl->tn', feats, w) + b[None, :]
        return pd.DataFrame(preds, index=prices.index, columns=self.target_cols_)
