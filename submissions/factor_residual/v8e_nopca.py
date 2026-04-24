"""v8e — No PCA at all. Per-asset mean-reversion on raw log-returns.
This tests whether the PCA decomposition is even needed."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from v3 import _load_target_pairs, _target_assets, _log_returns, LAGS


class Model:
    device = "cpu"
    L_R = 5
    alpha_R = 1.0

    def __init__(self) -> None:
        self.pairs_ = _load_target_pairs()
        self.assets_ = _target_assets(self.pairs_)
        self.target_cols_ = [f"target_{i}" for i in range(len(self.pairs_))]
        self.resid_coef_: dict[int, np.ndarray] = {}

    def _asset_prices(self, X):
        cols = [a for a in self.assets_ if a in X.columns]
        return X[cols].copy()

    def fit(self, X, y):
        prices = self._asset_prices(X).sort_index()
        R = _log_returns(prices).values  # (T, A) - used as "residual" here
        T, A = R.shape
        for lag in LAGS:
            if T - lag - 1 <= 0:
                continue
            # targets: sum of next lag daily returns
            tgt = np.zeros((T - lag, A))
            for i in range(1, lag + 1):
                tgt += R[i : T - lag + i]
            T_eff = T - lag
            feats_per_asset = np.zeros((T_eff, A, self.L_R))
            for t in range(T_eff):
                start = max(0, t - self.L_R + 1)
                block = R[start : t + 1]
                pad = self.L_R - block.shape[0]
                if pad > 0:
                    block = np.vstack([np.zeros((pad, A)), block])
                feats_per_asset[t] = block.T
            coefs = np.zeros((A, self.L_R + 1))
            for a in range(A):
                Xa = feats_per_asset[:, a, :]
                ya = tgt[:, a]
                r = Ridge(alpha=self.alpha_R)
                r.fit(Xa, ya)
                coefs[a, :self.L_R] = r.coef_
                coefs[a, -1] = r.intercept_
            self.resid_coef_[lag] = coefs

    def predict(self, X):
        prices = self._asset_prices(X).sort_index()
        R = _log_returns(prices).values
        T_test, A = R.shape
        feats = np.zeros((T_test, A, self.L_R))
        for t in range(T_test):
            start = max(0, t - self.L_R + 1)
            block = R[start : t + 1]
            pad = self.L_R - block.shape[0]
            if pad > 0:
                block = np.vstack([np.zeros((pad, A)), block])
            feats[t] = block.T

        asset_preds: dict[int, np.ndarray] = {}
        for lag in LAGS:
            coefs = self.resid_coef_.get(lag)
            if coefs is None:
                asset_preds[lag] = np.zeros((T_test, A))
                continue
            w = coefs[:, :self.L_R]
            b = coefs[:, -1]
            R_hat = np.einsum('tal,al->ta', feats, w) + b[None, :]
            asset_preds[lag] = R_hat

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
