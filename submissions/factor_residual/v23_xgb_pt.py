"""v23 — XGBoost per target. Same features as v10 (lagged 1-day target
values) but non-linear. Shallow trees, small count to keep runtime modest."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import xgboost as xgb

from v3 import _load_target_pairs, _target_assets, _log_returns, LAGS


class Model:
    device = "cpu"
    L = 5
    n_estimators = 50
    max_depth = 3
    learning_rate = 0.1
    reg_lambda = 1.0
    cpus_per_fold = 16

    def __init__(self):
        self.pairs_ = _load_target_pairs()
        self.assets_ = _target_assets(self.pairs_)
        self.target_cols_ = [f"target_{i}" for i in range(len(self.pairs_))]
        self.models_: list[xgb.XGBRegressor | None] = []

    def _asset_prices(self, X):
        cols = [a for a in self.assets_ if a in X.columns]
        return X[cols].copy()

    def _target_values(self, R):
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

    def _stack_lagged(self, V):
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

    def fit(self, X, y):
        prices = self._asset_prices(X).sort_index()
        R = _log_returns(prices).values
        V = self._target_values(R)
        feats = self._stack_lagged(V)
        T, N, D = feats.shape
        self.models_ = [None] * N
        for n in range(N):
            lag = int(self.pairs_.iloc[n]["lag"])
            if T - lag - 1 <= 0:
                continue
            tgt = np.zeros(T - lag)
            for i in range(1, lag + 1):
                tgt += V[i : T - lag + i, n]
            Xn = feats[: T - lag, n, :]
            m = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                reg_lambda=self.reg_lambda,
                tree_method="hist",
                n_jobs=1,  # per-target parallel across targets would help but
                # xgboost already scales; keep simple
                verbosity=0,
            )
            m.fit(Xn, tgt)
            self.models_[n] = m

    def predict(self, X):
        prices = self._asset_prices(X).sort_index()
        R = _log_returns(prices).values
        V = self._target_values(R)
        feats = self._stack_lagged(V)
        T_test, N, _ = feats.shape
        preds = np.zeros((T_test, N))
        for n in range(N):
            m = self.models_[n]
            if m is None:
                continue
            preds[:, n] = m.predict(feats[:, n, :])
        return pd.DataFrame(preds, index=prices.index, columns=self.target_cols_)
