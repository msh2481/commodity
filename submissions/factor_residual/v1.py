"""v1 — Pipeline sanity for factor + residual approach.

K=5 PCA on log-returns of the 106 target-underlying assets. Factor returns at
each horizon (lag ∈ 1..4) predicted by ridge from today's factor values.
Residual prediction set to zero. Target predictions reconstructed from factor
predictions via W.T, then composed per `target_pairs.csv` (single or spread).
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
    # prices: indexed by date_id, columns = assets.
    # Replace 0/negative with NaN (log undefined), then take diff of logs.
    p = prices.where(prices > 0)
    logp = np.log(p)
    r = logp.diff()
    # For the first row (and any gaps) fill with 0 — no move that day.
    return r.fillna(0.0)


class Model:
    device = "cpu"
    K = 5
    ridge_alpha = 1.0

    def __init__(self) -> None:
        self.pairs_ = _load_target_pairs()
        self.assets_ = _target_assets(self.pairs_)
        self.target_cols_ = [f"target_{i}" for i in range(len(self.pairs_))]
        self.pca_: PCA | None = None
        self.W_: np.ndarray | None = None  # (n_assets, K)
        self.ridges_: dict[int, Ridge] = {}

    def _asset_prices(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [a for a in self.assets_ if a in X.columns]
        return X[cols].copy()

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        prices = self._asset_prices(X).sort_index()
        R = _log_returns(prices)  # (T, A) daily log-returns

        self.pca_ = PCA(n_components=self.K, svd_solver="full")
        F = self.pca_.fit_transform(R.values)  # (T, K)
        self.W_ = self.pca_.components_.T  # (A, K), PCA returns (K, A)

        # Build per-horizon factor-return targets: sum of next ℓ daily F values.
        # Train ridge F_t -> sum_{i=1..ℓ} F_{t+i}.
        T = F.shape[0]
        for lag in LAGS:
            # target[t] = F[t+1] + ... + F[t+lag], valid for t in [0, T-lag-1]
            if T - lag - 1 <= 0:
                continue
            feats = F[: T - lag]  # rows 0..T-lag-1
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
        F_test = self.pca_.transform(R.values)  # (T_test, K)

        # For each test date, predict ℓ-step factor return, then per-asset return.
        T_test = F_test.shape[0]
        # per-lag asset return predictions: (T_test, A)
        asset_preds: dict[int, np.ndarray] = {}
        for lag in LAGS:
            ridge = self.ridges_.get(lag)
            if ridge is None:
                asset_preds[lag] = np.zeros((T_test, len(self.assets_)))
                continue
            F_hat = ridge.predict(F_test)  # (T_test, K)
            R_hat = F_hat @ self.W_.T  # (T_test, A)
            asset_preds[lag] = R_hat

        # Compose targets.
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
                a = pair.strip()
                preds[:, t_idx] = R_hat[:, asset_to_idx[a]]

        return pd.DataFrame(preds, index=prices.index, columns=self.target_cols_)
