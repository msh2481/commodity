"""v12 — Per-target ridge with richer features.
For spread target A-B: features = [V_target history, R_A history, R_B history]
For single-asset target A: features = [R_A history]
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
        self.target_models_: list[tuple[Ridge | None, list[int]]] = []  # per (target, lag)

    def _asset_prices(self, X):
        cols = [a for a in self.assets_ if a in X.columns]
        return X[cols].copy()

    def _features_for_target(self, R: np.ndarray, idx: int) -> np.ndarray:
        """Build feature matrix (T, D) for target idx.
        D depends on whether it's single (L features) or spread (3L features:
        R_A, R_B, spread = R_A - R_B)."""
        asset_to_idx = {a: i for i, a in enumerate(self.assets_)}
        row = self.pairs_.iloc[idx]
        pair = row["pair"]
        T = R.shape[0]
        if " - " in pair:
            a, b = [s.strip() for s in pair.split(" - ")]
            ra = R[:, asset_to_idx[a]]
            rb = R[:, asset_to_idx[b]]
            sp = ra - rb
            cols = [ra, rb, sp]
        else:
            a = pair.strip()
            cols = [R[:, asset_to_idx[a]]]
        # Stack lagged
        D = len(cols) * self.L
        feats = np.zeros((T, D))
        for ci, col in enumerate(cols):
            for l in range(self.L):
                # feature is col[t - l] for t in [0, T), zero-padded for t<l
                shifted = np.zeros(T)
                shifted[l:] = col[: T - l]
                feats[:, ci * self.L + l] = shifted
        return feats

    def _target_value_1d(self, R: np.ndarray, idx: int) -> np.ndarray:
        row = self.pairs_.iloc[idx]
        pair = row["pair"]
        asset_to_idx = {a: i for i, a in enumerate(self.assets_)}
        if " - " in pair:
            a, b = [s.strip() for s in pair.split(" - ")]
            return R[:, asset_to_idx[a]] - R[:, asset_to_idx[b]]
        return R[:, asset_to_idx[pair.strip()]]

    def fit(self, X, y):
        prices = self._asset_prices(X).sort_index()
        R = _log_returns(prices).values
        T = R.shape[0]
        N = len(self.pairs_)
        self.target_models_ = [None] * N

        # Precompute target values (1-day) for all targets
        V_all = np.zeros((T, N))
        for n in range(N):
            V_all[:, n] = self._target_value_1d(R, n)

        for n in range(N):
            lag = int(self.pairs_.iloc[n]["lag"])
            if T - lag - 1 <= 0:
                continue
            feats = self._features_for_target(R, n)  # (T, D)
            # target at date t = sum_{i=1..lag} V[t+i, n]
            tgt = np.zeros(T - lag)
            for i in range(1, lag + 1):
                tgt += V_all[i : T - lag + i, n]
            feat_train = feats[: T - lag]
            r = Ridge(alpha=self.alpha)
            r.fit(feat_train, tgt)
            self.target_models_[n] = r

    def predict(self, X):
        prices = self._asset_prices(X).sort_index()
        R = _log_returns(prices).values
        T_test = R.shape[0]
        N = len(self.pairs_)
        preds = np.zeros((T_test, N))
        for n in range(N):
            m = self.target_models_[n]
            if m is None:
                continue
            feats = self._features_for_target(R, n)
            preds[:, n] = m.predict(feats)
        return pd.DataFrame(preds, index=prices.index, columns=self.target_cols_)
