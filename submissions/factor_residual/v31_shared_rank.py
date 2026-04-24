"""v31 — Shared ridge across all (target, date) samples with cross-sectional
rank features and rank target. One set of coefficients applied to every
target; the rank normalization makes that fair across targets.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from v3 import _load_target_pairs, _target_assets, _log_returns, LAGS
from v28_pt_rank_xy import _xs_rank


class Model:
    device = "cpu"
    L = 5
    alpha = 1.0

    def __init__(self):
        self.pairs_ = _load_target_pairs()
        self.assets_ = _target_assets(self.pairs_)
        self.target_cols_ = [f"target_{i}" for i in range(len(self.pairs_))]
        # one ridge per lag (lag-specific coefficients)
        self.ridges_: dict[int, Ridge] = {}

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

    def _build_feats_ranked(self, V):
        """(T, N, L) of cross-sectionally rank-normalized lagged V."""
        T, N = V.shape
        feats = np.zeros((T, N, self.L))
        for l in range(self.L):
            shifted = np.zeros((T, N))
            shifted[l:] = V[:T - l]
            feats[:, :, l] = np.nan_to_num(_xs_rank(shifted), nan=0.0)
        return feats

    def fit(self, X, y):
        prices = self._asset_prices(X).sort_index()
        R = _log_returns(prices).values
        V = self._target_values(R)
        feats = self._build_feats_ranked(V)  # (T, N, L)
        T, N, D = feats.shape
        # Precompute each lag's mask of target indices
        lag_targets = {lag: [] for lag in LAGS}
        for n in range(N):
            lag_targets[int(self.pairs_.iloc[n]["lag"])].append(n)

        for lag in LAGS:
            targets = lag_targets[lag]
            if not targets or T - lag - 1 <= 0:
                continue
            targets = np.array(targets)
            # Build reconstructed label for all targets at this lag.
            labels_lag = np.full((T, N), np.nan)
            for n in targets:
                tgt = np.zeros(T - lag)
                for i in range(1, lag + 1):
                    tgt += V[i : T - lag + i, n]
                labels_lag[: T - lag, n] = tgt
            y_rank = _xs_rank(labels_lag)

            # Stack (date, target) samples, features = feats[t, n, :]
            # Use only targets of this lag and valid rows.
            stacked_X = []
            stacked_y = []
            for n in targets:
                mask = ~np.isnan(y_rank[:, n])
                if mask.sum() == 0:
                    continue
                stacked_X.append(feats[mask, n, :])
                stacked_y.append(y_rank[mask, n])
            if not stacked_X:
                continue
            Xs = np.vstack(stacked_X)
            ys = np.concatenate(stacked_y)
            r = Ridge(alpha=self.alpha)
            r.fit(Xs, ys)
            self.ridges_[lag] = r

    def predict(self, X):
        prices = self._asset_prices(X).sort_index()
        R = _log_returns(prices).values
        V = self._target_values(R)
        feats = self._build_feats_ranked(V)
        T_test, N, D = feats.shape
        preds = np.zeros((T_test, N))
        lag_to_targets = {lag: [] for lag in LAGS}
        for n in range(N):
            lag_to_targets[int(self.pairs_.iloc[n]["lag"])].append(n)
        for lag, targets in lag_to_targets.items():
            r = self.ridges_.get(lag)
            if r is None:
                continue
            # Predict per target
            for n in targets:
                preds[:, n] = r.predict(feats[:, n, :])
        return pd.DataFrame(preds, index=prices.index, columns=self.target_cols_)
