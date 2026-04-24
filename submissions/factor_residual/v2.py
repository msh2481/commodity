"""v2 — Richer factor features.

Same PCA as v1 (K=5, on 106 target assets' log-returns). Factor prediction
features = concatenation of last L=5 daily factor values (5*K features).
Residual prediction still zero.
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


def _build_features(F: np.ndarray, L: int) -> np.ndarray:
    """Stack last L factor-return rows (including t) into features per date.

    Row t of the output is [F[t-L+1], ..., F[t]]. Rows before L-1 are padded
    with zeros so every date has a feature row (important at the test-fold
    boundary).
    """
    T, K = F.shape
    feats = np.zeros((T, L * K))
    for t in range(T):
        start = max(0, t - L + 1)
        block = F[start : t + 1]
        pad = L - block.shape[0]
        if pad > 0:
            pad_block = np.zeros((pad, K))
            block = np.vstack([pad_block, block])
        feats[t] = block.reshape(-1)
    return feats


class Model:
    device = "cpu"
    K = 5
    L = 5  # factor history window
    ridge_alpha = 10.0

    def __init__(self) -> None:
        self.pairs_ = _load_target_pairs()
        self.assets_ = _target_assets(self.pairs_)
        self.target_cols_ = [f"target_{i}" for i in range(len(self.pairs_))]
        self.pca_: PCA | None = None
        self.W_: np.ndarray | None = None
        self.ridges_: dict[int, Ridge] = {}

    def _asset_prices(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [a for a in self.assets_ if a in X.columns]
        return X[cols].copy()

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        prices = self._asset_prices(X).sort_index()
        R = _log_returns(prices)

        self.pca_ = PCA(n_components=self.K, svd_solver="full")
        F = self.pca_.fit_transform(R.values)  # (T, K)
        self.W_ = self.pca_.components_.T  # (A, K)

        feats_all = _build_features(F, self.L)  # (T, L*K)
        T = F.shape[0]
        for lag in LAGS:
            if T - lag - 1 <= 0:
                continue
            feats = feats_all[: T - lag]
            tgts = np.zeros((T - lag, self.K))
            for i in range(1, lag + 1):
                tgts += F[i : T - lag + i]
            ridge = Ridge(alpha=self.ridge_alpha)
            ridge.fit(feats, tgts)
            self.ridges_[lag] = ridge

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.pca_ is not None and self.W_ is not None
        prices = self._asset_prices(X).sort_index()
        R = _log_returns(prices)
        F_test = self.pca_.transform(R.values)
        feats_test = _build_features(F_test, self.L)

        T_test = F_test.shape[0]
        asset_preds: dict[int, np.ndarray] = {}
        for lag in LAGS:
            ridge = self.ridges_.get(lag)
            if ridge is None:
                asset_preds[lag] = np.zeros((T_test, len(self.assets_)))
                continue
            F_hat = ridge.predict(feats_test)
            asset_preds[lag] = F_hat @ self.W_.T

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
