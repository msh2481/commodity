"""v43: multi-quantile XGBoost with expected-rank aggregation.

Motivation — the competition metric is per-date Spearman across 424 targets,
so only the *ordering* of predictions on each date matters. When each target's
predictive distribution is heteroscedastic / skewed, the score s_i that
maximises expected rank correlation is not the conditional mean:

    s_i* = E[rank(Y_i)] = 1 + Σ_{j≠i} P(Y_j < Y_i)

Given quantile regressors for each target we have an empirical CDF per target
and can compute s_i* in closed form (globally rank all N*M quantile samples
per date, then take the row-mean over the M samples of each target).

Base hparams: trial 350 from the session-2 skopt search (md=3 BO leader).
Feature pipeline and K-trick target mask: identical to v38_tune / v41.

Concrete subclasses set AGG_METHOD ∈ {"mean", "median", "erank"}.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import concurrent.futures
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import rankdata

from v38_tune import (
    _build_common, _build_caches,
    LME_COLS, JPX_MAIN, SPREAD_WINDOWS, N_THREADS_FIT, TARGET_PAIRS_CSV,
)


# Trial 350 (md=3 BO leader, overall Sharpe 0.2348 with L2 loss).
BASE_HPARAMS = dict(
    n_estimators=446,
    max_depth=3,
    learning_rate=0.17272761514595722,
    min_child_weight=22,
    subsample=0.8923203127103814,
    colsample_bytree=0.45661441988940543,
    reg_lambda=0.03905920232899221,
    reg_alpha=0.0011677526930123602,
    gamma=0.0011211744998537942,
)

# 9 symmetric quantiles, median at index 4.
ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MEDIAN_IDX = ALPHAS.index(0.5)


def _build_qparams():
    fixed = dict(
        device="cuda",
        tree_method="hist",
        verbosity=0,
        n_jobs=1,
        objective="reg:quantileerror",
        quantile_alpha=list(ALPHAS),
    )
    return {**fixed, **BASE_HPARAMS}


def _fit_one_q(args):
    X_np, yt, params = args
    mask = np.isfinite(yt)
    if mask.sum() < 50:
        return None
    m = xgb.XGBRegressor(**params)
    m.fit(X_np[mask], yt[mask].astype(np.float32))
    return m


def _predict_one_q(args):
    X_np, m = args
    if m is None:
        return None
    p = m.predict(X_np)
    # Guarantee (n_samples, n_quantiles) even if XGBoost squeezes a 1-d output.
    if p.ndim == 1:
        p = p[:, None]
    return p.astype(np.float32)


def _aggregate(q_mat: np.ndarray, method: str) -> np.ndarray:
    """Aggregate per-date quantile predictions (D, T_kept, M) into a scalar
    score per (date, target) of shape (D, T_kept)."""
    if method == "mean":
        return q_mat.mean(axis=2)
    if method == "median":
        return q_mat[:, :, MEDIAN_IDX]
    if method == "erank":
        D, T, M = q_mat.shape
        out = np.empty((D, T), dtype=np.float32)
        for d in range(D):
            flat = q_mat[d].reshape(-1)  # (T*M,)
            # average-ranks so ties are symmetric; NaN-free by construction.
            ranks = rankdata(flat, method="average").reshape(T, M)
            out[d] = ranks.mean(axis=1).astype(np.float32)
        return out
    raise ValueError(f"unknown AGG_METHOD: {method}")


class Model:
    device = "cuda"
    AGG_METHOD = "mean"  # overridden by subclasses

    def __init__(self):
        self.HPARAMS = _build_qparams()
        self._target_cols = None
        self._models = []
        self._pairs_df = pd.read_csv(TARGET_PAIRS_CSV)
        self._pair_of = dict(zip(self._pairs_df["target"], self._pairs_df["pair"]))
        self._lag_of = dict(zip(self._pairs_df["target"], self._pairs_df["lag"].astype(int)))
        self._X_train = None
        self._keep_idx = None

    def _combine(self, test_X=None):
        if test_X is None:
            return self._X_train.sort_index()
        c = pd.concat([self._X_train, test_X])
        return c[~c.index.duplicated(keep="first")].sort_index()

    def _target_key(self, t):
        p = self._pair_of[t]; lag = self._lag_of[t]
        if " - " in p:
            a, b = [s.strip() for s in p.split(" - ")]
            return (a, b, lag)
        return (p.strip(), None, lag)

    def _is_kept(self, t):
        if self._lag_of[t] < 4:
            return False
        p = self._pair_of[t]
        legs = [s.strip() for s in p.split(" - ")] if " - " in p else [p.strip()]
        return any(leg in LME_COLS for leg in legs)

    def _features_for_target(self, t, common, asset_cache, spread_cache):
        a, b, lag = self._target_key(t)
        parts = [common]
        if b is not None:
            if (a, b, lag) in spread_cache:
                parts.append(spread_cache[(a, b, lag)])
            if a in asset_cache: parts.append(asset_cache[a])
            if b in asset_cache: parts.append(asset_cache[b])
        else:
            if a in asset_cache: parts.append(asset_cache[a])
        return np.hstack(parts) if len(parts) > 1 else parts[0]

    def fit(self, X, y):
        self._target_cols = list(y.columns)
        self._X_train = X.copy()
        self._keep_idx = np.array([j for j, t in enumerate(self._target_cols) if self._is_kept(t)])

        full_X = self._combine(None)
        idx, common = _build_common(full_X)
        asset_cache, spread_cache = _build_caches(full_X, self._pairs_df)
        pos = idx.get_indexer(y.index)
        common = common[pos]
        asset_cache = {k: v[pos] for k, v in asset_cache.items()}
        spread_cache = {k: v[pos] for k, v in spread_cache.items()}

        self._models = [None] * len(self._target_cols)
        jobs = []; idx_for_jobs = []
        for j in self._keep_idx:
            t = self._target_cols[j]
            X_j = self._features_for_target(t, common, asset_cache, spread_cache)
            jobs.append((X_j, y[t].values, self.HPARAMS))
            idx_for_jobs.append(j)

        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
            results = list(ex.map(_fit_one_q, jobs))
        for j, m in zip(idx_for_jobs, results):
            self._models[j] = m

    def predict(self, X):
        full_X = self._combine(X)
        idx, common = _build_common(full_X)
        asset_cache, spread_cache = _build_caches(full_X, self._pairs_df)
        pos = idx.get_indexer(X.index)
        common = common[pos]
        asset_cache = {k: v[pos] for k, v in asset_cache.items()}
        spread_cache = {k: v[pos] for k, v in spread_cache.items()}

        jobs = []; idx_for_jobs = []
        for j in self._keep_idx:
            t = self._target_cols[j]
            X_j = self._features_for_target(t, common, asset_cache, spread_cache)
            jobs.append((X_j, self._models[j]))
            idx_for_jobs.append(j)

        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
            results = list(ex.map(_predict_one_q, jobs))

        D = len(X); T = len(self._target_cols); M = len(ALPHAS)
        q_mat = np.full((D, len(idx_for_jobs), M), np.nan, dtype=np.float32)
        for slot, p in enumerate(results):
            if p is not None:
                q_mat[:, slot, :] = p

        # If any target's entire quantile vector is NaN on some date, fall back
        # to the cross-section mean for that cell so erank/median still works.
        any_missing = np.all(np.isnan(q_mat), axis=2)  # (D, T_kept)
        if any_missing.any():
            valid_means = np.nanmean(q_mat, axis=2)  # (D, T_kept)
            per_date_fallback = np.nanmean(valid_means, axis=1, keepdims=True)
            fill = np.where(np.isnan(valid_means), per_date_fallback, valid_means)
            q_mat[any_missing] = np.broadcast_to(fill[any_missing, None], (int(any_missing.sum()), M))

        kept_scores = _aggregate(q_mat, self.AGG_METHOD)  # (D, T_kept)

        # K-trick: fill non-kept columns with per-date median of kept scores.
        raw = np.full((D, T), np.nan, dtype=np.float32)
        for slot, j in enumerate(idx_for_jobs):
            raw[:, j] = kept_scores[:, slot]
        med = np.nanmedian(raw[:, self._keep_idx], axis=1, keepdims=True)
        out = np.broadcast_to(med, raw.shape).copy()
        out[:, self._keep_idx] = raw[:, self._keep_idx]
        out = np.nan_to_num(out, nan=0.0)
        return pd.DataFrame(out, index=X.index, columns=self._target_cols)
