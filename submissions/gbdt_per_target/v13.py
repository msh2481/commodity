"""v13: v1 LME features (good baseline) + per-target A/B asset features + spread.

Hypothesis: v1's LME features generalize well (small, robust). Adding
per-target asset features and spread features (small, focused) gives
each target's model the most-relevant info.

Features per target:
  - 4 LME closes (raw, like v1)
  - 4 LME 1-day log-returns (like v1)
  - per-target A: ret1, ret5, ret20, vol20  (4)
  - per-target B (if spread): ret1, ret5, ret20, vol20 (4)
  - per-target spread: log(A/B), dev20, z20 (3)
  - cal_dow, cal_wom, cal_trend (3)

Total: 22 features (spread targets), 18 (single targets).
v1-like hparams: 100 trees depth 4 lr 0.05.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import concurrent.futures
import numpy as np
import pandas as pd
import xgboost as xgb


TARGET_PAIRS_CSV = "/root/commodity/data/target_pairs.csv"
N_THREADS_FIT = 8

LME_COLS = ["LME_AH_Close", "LME_CA_Close", "LME_PB_Close", "LME_ZS_Close"]


def _diff_n(a, k):
    n = len(a)
    out = np.full(n, np.nan, dtype=np.float32)
    if k < n:
        out[k:] = a[k:] - a[:-k]
    return out


def _asset_min_feats(logp):
    """4 features per asset: ret1, ret5, ret20, vol20."""
    ret1 = _diff_n(logp, 1)
    ret5 = _diff_n(logp, 5)
    ret20 = _diff_n(logp, 20)
    vol20 = pd.Series(ret1).rolling(20, min_periods=10).std().values.astype(np.float32)
    return np.stack([ret1, ret5, ret20, vol20], axis=1)


def _spread_min(logp_a, logp_b):
    spread = (logp_a - logp_b).astype(np.float32)
    s = pd.Series(spread)
    sma = s.rolling(20, min_periods=10).mean().values.astype(np.float32)
    std = s.rolling(20, min_periods=10).std().values.astype(np.float32)
    dev = (spread - sma).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(std > 1e-9, dev / std, np.nan).astype(np.float32)
    return np.stack([spread, dev, z], axis=1)


def _calendar(index):
    d = index.to_numpy()
    return np.stack([(d % 5).astype(np.float32),
                     (d % 22).astype(np.float32),
                     (d / 2000.0).astype(np.float32)], axis=1)


def _prep(X, target_pairs):
    Xs = X.sort_index()
    prices = Xs.ffill()

    # LME baseline (4 raw + 4 ret1) — same for all targets
    lme_logp = np.log(prices[LME_COLS].clip(lower=1e-12).values.astype(np.float64))
    lme_ret1 = np.column_stack([_diff_n(lme_logp[:, i], 1) for i in range(lme_logp.shape[1])])
    lme_block = np.hstack([
        np.log(prices[LME_COLS].clip(lower=1e-12).values.astype(np.float32)),
        lme_ret1.astype(np.float32),
    ])

    assets, pairs = set(), set()
    for p in target_pairs["pair"]:
        if " - " in p:
            a, b = [s.strip() for s in p.split(" - ")]
            assets.update([a, b]); pairs.add((a, b))
        else:
            assets.add(p.strip())

    asset_cache = {}
    for a in assets:
        if a in prices.columns:
            logp = np.log(prices[a].clip(lower=1e-12).values.astype(np.float64))
            asset_cache[a] = _asset_min_feats(logp)

    pair_cache = {}
    for a, b in pairs:
        if a in prices.columns and b in prices.columns:
            lpa = np.log(prices[a].clip(lower=1e-12).values.astype(np.float64))
            lpb = np.log(prices[b].clip(lower=1e-12).values.astype(np.float64))
            pair_cache[(a, b)] = _spread_min(lpa, lpb)

    cal = _calendar(Xs.index)
    common_tail = np.hstack([lme_block, cal])  # 8 + 3 = 11
    return Xs.index, asset_cache, pair_cache, common_tail


def _feats_for_target(pair_str, asset_cache, pair_cache, common_tail):
    parts = []
    if " - " in pair_str:
        a, b = [s.strip() for s in pair_str.split(" - ")]
        if a in asset_cache:
            parts.append(asset_cache[a])
        if b in asset_cache:
            parts.append(asset_cache[b])
        if (a, b) in pair_cache:
            parts.append(pair_cache[(a, b)])
    else:
        a = pair_str.strip()
        if a in asset_cache:
            parts.append(asset_cache[a])
    parts.append(common_tail)
    return np.hstack(parts) if len(parts) > 1 else parts[0]


def _fit_one(args):
    X_np, yt, params = args
    mask = ~np.isnan(yt)
    if mask.sum() < 50:
        return None
    m = xgb.XGBRegressor(**params)
    m.fit(X_np[mask], yt[mask].astype(np.float32))
    return m


def _predict_one(args):
    X_np, m = args
    if m is None:
        return None
    return m.predict(X_np)


class Model:
    device = "cuda"

    def __init__(self) -> None:
        self._target_cols = None
        self._models = []
        self._pairs_df = pd.read_csv(TARGET_PAIRS_CSV)
        self._pair_of = dict(zip(self._pairs_df["target"], self._pairs_df["pair"]))
        self._X_train = None

    def _combine(self, test_X=None):
        if test_X is None:
            return self._X_train.sort_index()
        c = pd.concat([self._X_train, test_X])
        return c[~c.index.duplicated(keep="first")].sort_index()

    def fit(self, X, y):
        self._target_cols = list(y.columns)
        self._X_train = X.copy()
        full_X = self._combine(None)
        idx, ac, pc, ct = _prep(full_X, self._pairs_df)
        pos = idx.get_indexer(y.index)
        ac = {k: v[pos] for k, v in ac.items()}
        pc = {k: v[pos] for k, v in pc.items()}
        ct = ct[pos]
        params = dict(device="cuda", tree_method="hist", n_estimators=100,
                      max_depth=4, learning_rate=0.05, verbosity=0, n_jobs=1)
        jobs = [(_feats_for_target(self._pair_of[t], ac, pc, ct), y[t].values, params)
                for t in self._target_cols]
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
            self._models = list(ex.map(_fit_one, jobs))

    def predict(self, X):
        full_X = self._combine(X)
        idx, ac, pc, ct = _prep(full_X, self._pairs_df)
        pos = idx.get_indexer(X.index)
        ac = {k: v[pos] for k, v in ac.items()}
        pc = {k: v[pos] for k, v in pc.items()}
        ct = ct[pos]
        jobs = [(_feats_for_target(self._pair_of[t], ac, pc, ct), self._models[j])
                for j, t in enumerate(self._target_cols)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
            results = list(ex.map(_predict_one, jobs))
        out = np.zeros((len(X), len(self._target_cols)), dtype=np.float32)
        for j, p in enumerate(results):
            if p is not None:
                out[:, j] = p
        return pd.DataFrame(out, index=X.index, columns=self._target_cols)
