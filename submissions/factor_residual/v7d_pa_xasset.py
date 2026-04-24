"""Per-asset residual ridge with cross-asset features.

Features for asset a at time t:
  - own residual at t-L_R+1..t (L_R features)
  - today's residuals of ALL assets (A features)

This lets each asset exploit its specific dependencies on peers.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from v3 import _load_target_pairs, _target_assets, _log_returns, _stack_lagged, _next_lag_sum, LAGS


class Model:
    device = "cpu"
    K = 10
    L_F = 5
    L_R = 5
    alpha_F = 10.0
    alpha_R = 10.0

    def __init__(self) -> None:
        self.pairs_ = _load_target_pairs()
        self.assets_ = _target_assets(self.pairs_)
        self.target_cols_ = [f"target_{i}" for i in range(len(self.pairs_))]
        self.pca_: PCA | None = None
        self.W_: np.ndarray | None = None
        self.ridge_F_: dict[int, Ridge] = {}
        self.resid_coef_: dict[int, np.ndarray] = {}

    def _asset_prices(self, X):
        cols = [a for a in self.assets_ if a in X.columns]
        return X[cols].copy()

    def _build_resid_feats(self, eps: np.ndarray) -> np.ndarray:
        """Return (T, A, D_feat) where D_feat = L_R + A.
        Per-asset-a, per-date-t feature vector:
          [eps[a,t-L_R+1..t], eps[:, t]]
        """
        T, A = eps.shape
        feats = np.zeros((T, A, self.L_R + A))
        # own-lagged block
        for t in range(T):
            start = max(0, t - self.L_R + 1)
            block = eps[start : t + 1]  # (<=L_R, A)
            pad = self.L_R - block.shape[0]
            if pad > 0:
                block = np.vstack([np.zeros((pad, A)), block])
            # block.T is (A, L_R)
            feats[t, :, :self.L_R] = block.T
        # cross-asset today block — broadcast
        feats[:, :, self.L_R:] = eps[:, None, :]  # (T,1,A) -> (T,A,A) replicated
        return feats

    def fit(self, X, y):
        prices = self._asset_prices(X).sort_index()
        R = _log_returns(prices)
        self.pca_ = PCA(n_components=self.K, svd_solver="full")
        F = self.pca_.fit_transform(R.values)
        self.W_ = self.pca_.components_.T
        eps = R.values - F @ self.W_.T

        feats_F_all = _stack_lagged(F, self.L_F)
        T, A = eps.shape
        resid_feats_all = self._build_resid_feats(eps)  # (T, A, L_R + A)

        for lag in LAGS:
            if T - lag - 1 <= 0:
                continue
            tgt_F = _next_lag_sum(F, lag)
            tgt_R = _next_lag_sum(eps, lag)

            rf = Ridge(alpha=self.alpha_F)
            rf.fit(feats_F_all[: T - lag], tgt_F)
            self.ridge_F_[lag] = rf

            T_eff = T - lag
            feats = resid_feats_all[: T_eff]  # (T_eff, A, L_R+A)
            # per-asset ridge
            D = self.L_R + A
            coefs = np.zeros((A, D + 1))
            for a in range(A):
                Xa = feats[:, a, :]
                ya = tgt_R[:, a]
                r = Ridge(alpha=self.alpha_R)
                r.fit(Xa, ya)
                coefs[a, :D] = r.coef_
                coefs[a, -1] = r.intercept_
            self.resid_coef_[lag] = coefs

    def predict(self, X):
        prices = self._asset_prices(X).sort_index()
        R = _log_returns(prices)
        F_test = self.pca_.transform(R.values)
        eps_test = R.values - F_test @ self.W_.T

        feats_F = _stack_lagged(F_test, self.L_F)
        resid_feats = self._build_resid_feats(eps_test)  # (T_test, A, L_R+A)
        T_test, A, D = resid_feats.shape

        asset_preds: dict[int, np.ndarray] = {}
        for lag in LAGS:
            rf = self.ridge_F_.get(lag)
            coefs = self.resid_coef_.get(lag)
            if rf is None:
                asset_preds[lag] = np.zeros((T_test, A))
                continue
            F_hat = rf.predict(feats_F)
            R_hat_factor = F_hat @ self.W_.T

            eps_hat = np.zeros((T_test, A))
            if coefs is not None:
                w = coefs[:, :D]  # (A, D)
                b = coefs[:, -1]  # (A,)
                eps_hat = np.einsum('tad,ad->ta', resid_feats, w) + b[None, :]

            asset_preds[lag] = R_hat_factor + eps_hat

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
