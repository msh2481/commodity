"""v10 — Per-target ridge.
For each of the 424 targets, build a history of the target value (single-asset
log-return for single targets, spread return for spread targets), then train
a separate ridge per target using lagged target-value history as features.
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
    alpha = 1.0

    def __init__(self) -> None:
        self.pairs_ = _load_target_pairs()
        self.assets_ = _target_assets(self.pairs_)
        self.target_cols_ = [f"target_{i}" for i in range(len(self.pairs_))]
        # per-target linear coefs: (n_targets, L + 1)
        self.target_coefs_: np.ndarray | None = None

    def _asset_prices(self, X):
        cols = [a for a in self.assets_ if a in X.columns]
        return X[cols].copy()

    def _target_values(self, R: np.ndarray) -> np.ndarray:
        """Compute per-date value of each of the 424 targets.

        R: (T, A) log-returns of the 106 underlying assets.
        Returns: (T, n_targets) where col i is the 1-day value of target i.
        For spread targets this is R_A - R_B; for single it's R_A. The target's
        ℓ-step value used in the competition equals sum of ℓ consecutive daily
        values of this time series.
        """
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

    def _stack_lagged(self, V: np.ndarray, L: int) -> np.ndarray:
        """V: (T, N) -> out (T, N, L) with row t = V[t-L+1..t]^T."""
        T, N = V.shape
        out = np.zeros((T, N, L))
        for t in range(T):
            start = max(0, t - L + 1)
            block = V[start : t + 1]
            pad = L - block.shape[0]
            if pad > 0:
                block = np.vstack([np.zeros((pad, N)), block])
            out[t] = block.T
        return out

    def fit(self, X, y):
        prices = self._asset_prices(X).sort_index()
        R = _log_returns(prices).values
        V = self._target_values(R)  # (T, n_targets)
        T, N = V.shape
        feats_all = self._stack_lagged(V, self.L)  # (T, N, L)

        # We train a separate ridge per (target, lag). Since per-target ridge
        # has only L features, we can do it in closed form very fast using
        # precomputed Gram matrices. But sklearn Ridge is fine for 424 fits.
        coefs = np.zeros((len(LAGS), N, self.L + 1))
        for li, lag in enumerate(LAGS):
            if T - lag - 1 <= 0:
                continue
            # Target per date per target = sum of next lag values.
            tgt = np.zeros((T - lag, N))
            for i in range(1, lag + 1):
                tgt += V[i : T - lag + i]
            feats = feats_all[: T - lag]  # (T-lag, N, L)
            for n in range(N):
                Xn = feats[:, n, :]
                yn = tgt[:, n]
                r = Ridge(alpha=self.alpha)
                r.fit(Xn, yn)
                coefs[li, n, :self.L] = r.coef_
                coefs[li, n, -1] = r.intercept_
        self.target_coefs_ = coefs

    def predict(self, X):
        prices = self._asset_prices(X).sort_index()
        R = _log_returns(prices).values
        V = self._target_values(R)
        T_test, N = V.shape
        feats = self._stack_lagged(V, self.L)  # (T_test, N, L)

        preds = np.zeros((T_test, N))
        coefs = self.target_coefs_
        lag_to_row = {lag: i for i, lag in enumerate(LAGS)}
        for t_idx, row in self.pairs_.iterrows():
            lag = int(row["lag"])
            li = lag_to_row[lag]
            w = coefs[li, t_idx, :self.L]  # (L,)
            b = coefs[li, t_idx, -1]
            preds[:, t_idx] = (feats[:, t_idx, :] * w[None, :]).sum(axis=1) + b

        return pd.DataFrame(preds, index=prices.index, columns=self.target_cols_)
