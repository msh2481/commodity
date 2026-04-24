"""v9a — Volatility-normalized per-asset residual mean-reversion.
Hypothesis: reversion is proportional to |σ|; normalizing by rolling vol
gives cleaner cross-sectional ranking."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

from v3 import _load_target_pairs, _target_assets, _log_returns, LAGS


class Model:
    device = "cpu"
    K = 5  # PCA factors to project out
    L_R = 5
    vol_window = 20
    alpha_R = 1.0

    def __init__(self) -> None:
        self.pairs_ = _load_target_pairs()
        self.assets_ = _target_assets(self.pairs_)
        self.target_cols_ = [f"target_{i}" for i in range(len(self.pairs_))]
        self.pca_: PCA | None = None
        self.W_: np.ndarray | None = None
        self.resid_coef_: dict[int, np.ndarray] = {}

    def _asset_prices(self, X):
        cols = [a for a in self.assets_ if a in X.columns]
        return X[cols].copy()

    def _rolling_std(self, arr: np.ndarray) -> np.ndarray:
        """Lagged rolling std (excludes current day). Returns shape (T, A)
        with vol at time t computed from arr[max(0, t-W):t]."""
        T, A = arr.shape
        out = np.ones((T, A)) * 1e-4  # avoid zero divide
        for t in range(1, T):
            start = max(0, t - self.vol_window)
            block = arr[start:t]
            if block.shape[0] >= 2:
                s = block.std(axis=0, ddof=0)
                out[t] = np.where(s > 1e-6, s, 1e-4)
        return out

    def _build_feats(self, eps: np.ndarray, vol: np.ndarray) -> np.ndarray:
        """(T, A, L_R) of normalized recent residuals: eps[:, a] / vol[t, a]."""
        T, A = eps.shape
        feats = np.zeros((T, A, self.L_R))
        for t in range(T):
            start = max(0, t - self.L_R + 1)
            block = eps[start : t + 1]
            pad = self.L_R - block.shape[0]
            if pad > 0:
                block = np.vstack([np.zeros((pad, A)), block])
            feats[t] = block.T
        # normalize by vol at time t (so features are in σ units)
        feats = feats / vol[:, :, None]
        return feats

    def fit(self, X, y):
        prices = self._asset_prices(X).sort_index()
        R = _log_returns(prices).values
        # Project out factors
        self.pca_ = PCA(n_components=self.K, svd_solver="full")
        F = self.pca_.fit_transform(R)
        self.W_ = self.pca_.components_.T
        eps = R - F @ self.W_.T
        vol = self._rolling_std(eps)

        feats_all = self._build_feats(eps, vol)
        T, A = eps.shape
        for lag in LAGS:
            if T - lag - 1 <= 0:
                continue
            tgt = np.zeros((T - lag, A))
            for i in range(1, lag + 1):
                tgt += eps[i : T - lag + i]
            # Normalize targets by vol at t too (prediction will be un-normalized)
            tgt_norm = tgt / vol[: T - lag]
            feats = feats_all[: T - lag]
            coefs = np.zeros((A, self.L_R + 1))
            for a in range(A):
                Xa = feats[:, a, :]
                ya = tgt_norm[:, a]
                r = Ridge(alpha=self.alpha_R)
                r.fit(Xa, ya)
                coefs[a, :self.L_R] = r.coef_
                coefs[a, -1] = r.intercept_
            self.resid_coef_[lag] = coefs

    def predict(self, X):
        prices = self._asset_prices(X).sort_index()
        R = _log_returns(prices).values
        F_test = self.pca_.transform(R)
        eps = R - F_test @ self.W_.T
        vol = self._rolling_std(eps)
        feats = self._build_feats(eps, vol)

        T_test, A, _ = feats.shape
        asset_preds: dict[int, np.ndarray] = {}
        for lag in LAGS:
            coefs = self.resid_coef_.get(lag)
            if coefs is None:
                asset_preds[lag] = np.zeros((T_test, A))
                continue
            w = coefs[:, :self.L_R]
            b = coefs[:, -1]
            pred_norm = np.einsum('tal,al->ta', feats, w) + b[None, :]
            # Un-normalize: multiply by vol back
            asset_preds[lag] = pred_norm * vol

        n_targets = len(self.pairs_)
        preds = np.zeros((T_test, n_targets))
        asset_to_idx = {a: i for i, a in enumerate(self.assets_)}
        for t_idx, row in self.pairs_.iterrows():
            lag = int(row["lag"])
            pair = row["pair"]
            R_hat = asset_preds[lag]
            if " - " in pair:
                a, b = [s.strip() for s in pair.split(" - ")]
                preds[:, t_idx] = R_hat[:, asset_to_idx[a]] - R_hat[:, asset_to_idx[b]]
            else:
                preds[:, t_idx] = R_hat[:, asset_to_idx[pair.strip()]]
        return pd.DataFrame(preds, index=prices.index, columns=self.target_cols_)
