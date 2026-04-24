"""v3 — Factor + residual model.

PCA K=10 on 106 target-asset log-returns. Factor model: ridge on last L_F=5
factor histories predicts sum of next lag factor returns. Residual model:
shared ridge across assets learns mean-reversion pattern from last L_R=5
residual history (per-asset-ID stacked) to predict sum-of-next-lag residuals.
Recomposition: R_hat = F_hat @ W.T + eps_hat.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
LAGS = [1, 2, 3, 4]


def _load_target_pairs() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "target_pairs.csv")


def _target_assets(pairs: pd.DataFrame) -> list[str]:
    assets: set[str] = set()
    for p in pairs["pair"]:
        for a in p.split(" - "):
            assets.add(a.strip())
    return sorted(assets)


def _log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    p = prices.where(prices > 0)
    logp = np.log(p)
    r = logp.diff()
    return r.fillna(0.0)


def _stack_lagged(arr: np.ndarray, L: int) -> np.ndarray:
    """arr: (T, D). Output: (T, L*D) with row t = [arr[t-L+1], ..., arr[t]],
    zero-padded for t < L-1."""
    T, D = arr.shape
    out = np.zeros((T, L * D))
    for t in range(T):
        start = max(0, t - L + 1)
        block = arr[start : t + 1]
        pad = L - block.shape[0]
        if pad > 0:
            block = np.vstack([np.zeros((pad, D)), block])
        out[t] = block.reshape(-1)
    return out


def _next_lag_sum(arr: np.ndarray, lag: int) -> np.ndarray:
    """arr: (T, D). Out: (T - lag, D) where row t = arr[t+1] + ... + arr[t+lag]."""
    T = arr.shape[0]
    out = np.zeros((T - lag, arr.shape[1]))
    for i in range(1, lag + 1):
        out += arr[i : T - lag + i]
    return out


class Model:
    device = "cpu"
    K = 10
    L_F = 5
    L_R = 5
    alpha_F = 10.0
    alpha_R = 100.0

    def __init__(self) -> None:
        self.pairs_ = _load_target_pairs()
        self.assets_ = _target_assets(self.pairs_)
        self.target_cols_ = [f"target_{i}" for i in range(len(self.pairs_))]
        self.pca_: PCA | None = None
        self.W_: np.ndarray | None = None
        self.ridge_F_: dict[int, Ridge] = {}
        self.ridge_R_: dict[int, Ridge] = {}

    def _asset_prices(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [a for a in self.assets_ if a in X.columns]
        return X[cols].copy()

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        prices = self._asset_prices(X).sort_index()
        R = _log_returns(prices)  # (T, A)

        self.pca_ = PCA(n_components=self.K, svd_solver="full")
        F = self.pca_.fit_transform(R.values)  # (T, K)
        self.W_ = self.pca_.components_.T  # (A, K)
        eps = R.values - F @ self.W_.T  # (T, A)

        feats_F_all = _stack_lagged(F, self.L_F)  # (T, L_F*K)
        feats_R_all = _stack_lagged(eps, self.L_R)  # (T, L_R*A)

        T, A = eps.shape
        for lag in LAGS:
            if T - lag - 1 <= 0:
                continue
            tgt_F = _next_lag_sum(F, lag)  # (T-lag, K)
            tgt_R = _next_lag_sum(eps, lag)  # (T-lag, A)

            rf = Ridge(alpha=self.alpha_F)
            rf.fit(feats_F_all[: T - lag], tgt_F)
            self.ridge_F_[lag] = rf

            # Residual model: per-asset, shared-coefficient ridge.
            # For each (asset a, date t), predictor = residual history of a
            # up to t (L_R lags, 1 feature per lag). Target = sum-of-next-lag
            # residuals for a. Stack across assets.
            T_eff = T - lag
            # Build (T_eff * A, L_R) features and (T_eff * A, 1) targets
            # by stacking per-asset residual histories.
            feats_per_asset = np.zeros((T_eff, A, self.L_R))
            for t in range(T_eff):
                start = max(0, t - self.L_R + 1)
                block = eps[start : t + 1]  # (<=L_R, A)
                pad = self.L_R - block.shape[0]
                if pad > 0:
                    block = np.vstack([np.zeros((pad, A)), block])
                # block shape (L_R, A); we want (A, L_R)
                feats_per_asset[t] = block.T
            X_stack = feats_per_asset.reshape(-1, self.L_R)
            y_stack = tgt_R.reshape(-1)
            rr = Ridge(alpha=self.alpha_R)
            rr.fit(X_stack, y_stack)
            self.ridge_R_[lag] = rr

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.pca_ is not None and self.W_ is not None
        prices = self._asset_prices(X).sort_index()
        R = _log_returns(prices)
        F_test = self.pca_.transform(R.values)
        eps_test = R.values - F_test @ self.W_.T

        feats_F = _stack_lagged(F_test, self.L_F)
        # Residual features per (date, asset): last L_R residuals.
        T_test, A = eps_test.shape
        feats_R_per_asset = np.zeros((T_test, A, self.L_R))
        for t in range(T_test):
            start = max(0, t - self.L_R + 1)
            block = eps_test[start : t + 1]
            pad = self.L_R - block.shape[0]
            if pad > 0:
                block = np.vstack([np.zeros((pad, A)), block])
            feats_R_per_asset[t] = block.T

        asset_preds: dict[int, np.ndarray] = {}
        for lag in LAGS:
            rf = self.ridge_F_.get(lag)
            rr = self.ridge_R_.get(lag)
            if rf is None:
                asset_preds[lag] = np.zeros((T_test, A))
                continue
            F_hat = rf.predict(feats_F)
            R_hat_factor = F_hat @ self.W_.T

            eps_hat = np.zeros((T_test, A))
            if rr is not None:
                Xr = feats_R_per_asset.reshape(-1, self.L_R)
                eps_hat = rr.predict(Xr).reshape(T_test, A)

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
