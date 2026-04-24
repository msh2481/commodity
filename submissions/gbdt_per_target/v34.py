"""v34: v15 + minimal per-target asset features (just ret5 + vol20 of A and B).

Hypothesis: v13 added 4 per-asset features (ret1, ret5, ret20, vol20) × 2 assets = 8
and scored 0.178 (worse than v14). The failure mode was "too many features dilute."
Try just 2 per asset (ret5, vol20) = 4 total extra. Minimal and targeted.

Total features for spread targets: 8 LME + 3 cal + 7 spread + 4 asset = 22.
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import concurrent.futures
import numpy as np
import pandas as pd
import xgboost as xgb


TARGET_PAIRS_CSV = "/root/commodity/data/target_pairs.csv"
N_THREADS_FIT = 8

LME_COLS = ["LME_AH_Close", "LME_CA_Close", "LME_PB_Close", "LME_ZS_Close"]
SPREAD_WINDOWS = (5, 20, 60)


def _asset_min(logp):
    """2 features per asset: ret5, vol20(ret1)."""
    n = len(logp); ret5 = np.full(n, np.nan, dtype=np.float32)
    if 5 < n: ret5[5:] = (logp[5:] - logp[:-5]).astype(np.float32)
    ret1 = np.full(n, np.nan, dtype=np.float32)
    if 1 < n: ret1[1:] = (logp[1:] - logp[:-1]).astype(np.float32)
    vol20 = pd.Series(ret1).rolling(20, min_periods=10).std().values.astype(np.float32)
    return np.stack([ret5, vol20], axis=1)


def _spread_multi(logp_a, logp_b):
    spread = (logp_a - logp_b).astype(np.float32)
    s = pd.Series(spread)
    feats = [spread]
    for w in SPREAD_WINDOWS:
        sma = s.rolling(w, min_periods=max(3, w // 2)).mean().values.astype(np.float32)
        std = s.rolling(w, min_periods=max(3, w // 2)).std().values.astype(np.float32)
        dev = (spread - sma).astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(std > 1e-9, dev / std, np.nan).astype(np.float32)
        feats.extend([dev, z])
    return np.stack(feats, axis=1)


def _build_lme(X):
    Xs = X.sort_index(); prices = Xs[LME_COLS].ffill()
    logp = np.log(prices.clip(lower=1e-9)); ret1 = logp.diff()
    feats = pd.concat([prices, ret1.add_suffix("_ret1")], axis=1).fillna(0.0).astype(np.float32)
    return Xs.index, feats.values


def _calendar(index):
    d = index.to_numpy()
    return np.stack([(d % 5).astype(np.float32), (d % 22).astype(np.float32),
                     (d / 2000.0).astype(np.float32)], axis=1)


def _build_caches(X, target_pairs):
    Xs = X.sort_index(); prices = Xs.ffill()
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
            lp = np.log(prices[a].clip(lower=1e-12).values.astype(np.float64))
            asset_cache[a] = _asset_min(lp)
    spread_cache = {}
    for a, b in pairs:
        if a in prices.columns and b in prices.columns:
            lpa = np.log(prices[a].clip(lower=1e-12).values.astype(np.float64))
            lpb = np.log(prices[b].clip(lower=1e-12).values.astype(np.float64))
            spread_cache[(a, b)] = _spread_multi(lpa, lpb)
    return asset_cache, spread_cache


def _fit_one(args):
    X_np, yt, params = args
    mask = ~np.isnan(yt)
    if mask.sum() < 50: return None
    m = xgb.XGBRegressor(**params); m.fit(X_np[mask], yt[mask].astype(np.float32))
    return m


def _predict_one(args):
    X_np, m = args
    if m is None: return None
    return m.predict(X_np)


class Model:
    device = "cuda"
    def __init__(self):
        self._target_cols = None; self._models = []
        self._pairs_df = pd.read_csv(TARGET_PAIRS_CSV)
        self._pair_of = dict(zip(self._pairs_df["target"], self._pairs_df["pair"]))
        self._X_train = None
    def _combine(self, test_X=None):
        if test_X is None: return self._X_train.sort_index()
        c = pd.concat([self._X_train, test_X])
        return c[~c.index.duplicated(keep="first")].sort_index()
    def _pair_tuple(self, t):
        p = self._pair_of[t]
        if " - " in p:
            return tuple(s.strip() for s in p.split(" - "))
        return (p.strip(), None)
    def fit(self, X, y):
        self._target_cols = list(y.columns); self._X_train = X.copy()
        full_X = self._combine(None)
        idx, lme = _build_lme(full_X); cal = _calendar(idx)
        asset_cache, spread_cache = _build_caches(full_X, self._pairs_df)
        pos = idx.get_indexer(y.index)
        lme = lme[pos]; cal = cal[pos]
        asset_cache = {k: v[pos] for k, v in asset_cache.items()}
        spread_cache = {k: v[pos] for k, v in spread_cache.items()}
        common = np.hstack([lme, cal])

        params = dict(device="cuda", tree_method="hist", n_estimators=100,
                      max_depth=4, learning_rate=0.05, verbosity=0, n_jobs=1)
        jobs = []
        for t in self._target_cols:
            a, b = self._pair_tuple(t)
            parts = [common]
            if (a, b) in spread_cache:
                parts.append(spread_cache[(a, b)])
            if a in asset_cache:
                parts.append(asset_cache[a])
            if b is not None and b in asset_cache:
                parts.append(asset_cache[b])
            X_j = np.hstack(parts)
            jobs.append((X_j, y[t].values, params))
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
            self._models = list(ex.map(_fit_one, jobs))
    def predict(self, X):
        full_X = self._combine(X)
        idx, lme = _build_lme(full_X); cal = _calendar(idx)
        asset_cache, spread_cache = _build_caches(full_X, self._pairs_df)
        pos = idx.get_indexer(X.index)
        lme = lme[pos]; cal = cal[pos]
        asset_cache = {k: v[pos] for k, v in asset_cache.items()}
        spread_cache = {k: v[pos] for k, v in spread_cache.items()}
        common = np.hstack([lme, cal])

        jobs = []
        for j, t in enumerate(self._target_cols):
            a, b = self._pair_tuple(t)
            parts = [common]
            if (a, b) in spread_cache:
                parts.append(spread_cache[(a, b)])
            if a in asset_cache:
                parts.append(asset_cache[a])
            if b is not None and b in asset_cache:
                parts.append(asset_cache[b])
            X_j = np.hstack(parts)
            jobs.append((X_j, self._models[j]))
        out = np.zeros((len(X), len(self._target_cols)), dtype=np.float32)
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
            results = list(ex.map(_predict_one, jobs))
        for j, p in enumerate(results):
            if p is not None: out[:, j] = p
        return pd.DataFrame(out, index=X.index, columns=self._target_cols)
