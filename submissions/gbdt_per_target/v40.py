"""v40: ensemble of tuned configs from skopt search.

Reads top-K diverse hparam sets from _tune/ensemble_configs.json (list of dicts),
fits one model per config × per target, averages predictions. Same features as
v38_tune (reduced LME, no ret5/ret20/vol20). Applies lag>=4 & has_LME K-trick.
"""
from __future__ import annotations

import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import concurrent.futures
import numpy as np
import pandas as pd
import xgboost as xgb

from v38_tune import (
    _build_common, _build_caches, _spread_features, _asset_min,
    _fit_one, _predict_one,
    LME_COLS, JPX_MAIN, SPREAD_WINDOWS, N_THREADS_FIT, TARGET_PAIRS_CSV,
)

ENSEMBLE_CONFIGS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "_tune", "ensemble_configs.json")


def _load_configs():
    with open(ENSEMBLE_CONFIGS_JSON) as f:
        confs = json.load(f)
    fixed = dict(device="cuda", tree_method="hist", verbosity=0, n_jobs=1)
    return [{**fixed, **c} for c in confs]


class Model:
    device = "cuda"

    def __init__(self):
        self.CONFIGS = _load_configs()
        self._target_cols = None
        self._models_per_cfg = []
        self._pairs_df = pd.read_csv(TARGET_PAIRS_CSV)
        self._pair_of = dict(zip(self._pairs_df["target"], self._pairs_df["pair"]))
        self._lag_of = dict(zip(self._pairs_df["target"], self._pairs_df["lag"].astype(int)))
        self._X_train = None
        self._keep_idx = None

    def _combine(self, test_X=None):
        if test_X is None: return self._X_train.sort_index()
        c = pd.concat([self._X_train, test_X])
        return c[~c.index.duplicated(keep="first")].sort_index()

    def _target_key(self, t):
        p = self._pair_of[t]; lag = self._lag_of[t]
        if " - " in p:
            a, b = [s.strip() for s in p.split(" - ")]
            return (a, b, lag)
        return (p.strip(), None, lag)

    def _is_kept(self, t):
        if self._lag_of[t] < 4: return False
        p = self._pair_of[t]
        if " - " in p:
            legs = [s.strip() for s in p.split(" - ")]
        else:
            legs = [p.strip()]
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

        # Train one model per (config, target) combination
        self._models_per_cfg = [[None] * len(self._target_cols) for _ in self.CONFIGS]
        for cfg_i, cfg in enumerate(self.CONFIGS):
            jobs = []; idx_for_jobs = []
            for j in self._keep_idx:
                t = self._target_cols[j]
                X_j = self._features_for_target(t, common, asset_cache, spread_cache)
                jobs.append((X_j, y[t].values, cfg))
                idx_for_jobs.append(j)
            with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
                results = list(ex.map(_fit_one, jobs))
            for j, m in zip(idx_for_jobs, results):
                self._models_per_cfg[cfg_i][j] = m

    def predict(self, X):
        full_X = self._combine(X)
        idx, common = _build_common(full_X)
        asset_cache, spread_cache = _build_caches(full_X, self._pairs_df)
        pos = idx.get_indexer(X.index)
        common = common[pos]
        asset_cache = {k: v[pos] for k, v in asset_cache.items()}
        spread_cache = {k: v[pos] for k, v in spread_cache.items()}

        D = len(X); T = len(self._target_cols)
        raw_sum = np.zeros((D, T), dtype=np.float64)
        raw_cnt = np.zeros((D, T), dtype=np.int32)

        for models in self._models_per_cfg:
            jobs = []; idx_for_jobs = []
            for j in self._keep_idx:
                t = self._target_cols[j]
                X_j = self._features_for_target(t, common, asset_cache, spread_cache)
                jobs.append((X_j, models[j]))
                idx_for_jobs.append(j)
            with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
                results = list(ex.map(_predict_one, jobs))
            for j, p in zip(idx_for_jobs, results):
                if p is not None:
                    raw_sum[:, j] += p
                    raw_cnt[:, j] += 1

        raw = np.full((D, T), np.nan, dtype=np.float32)
        valid = raw_cnt > 0
        raw[valid] = (raw_sum[valid] / raw_cnt[valid]).astype(np.float32)

        kept = raw[:, self._keep_idx]
        med = np.nanmedian(kept, axis=1, keepdims=True)
        out = np.broadcast_to(med, raw.shape).copy()
        out[:, self._keep_idx] = raw[:, self._keep_idx]
        out = np.nan_to_num(out, nan=0.0)
        return pd.DataFrame(out, index=X.index, columns=self._target_cols)
