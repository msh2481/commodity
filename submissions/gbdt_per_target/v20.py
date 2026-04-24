"""v20: v15 + JPX Gold/Platinum baseline features.

Adds 4 JPX close columns and their 1-day log returns to the common tail
(shared across all targets). Rationale: many targets involve JPX Gold
Standard or Platinum Standard; these features should help per-target models
pick up JPX regime info even though the specific A or B may not be JPX.

Total common tail: 8 LME + 3 cal + 4 JPX closes + 4 JPX 1d ret = 19.
Plus per-target spread features (multi-window).
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
JPX_COLS = [
    "JPX_Gold_Standard_Futures_Close",
    "JPX_Platinum_Standard_Futures_Close",
    "JPX_Gold_Mini_Futures_Close",
    "JPX_Platinum_Mini_Futures_Close",
]
SPREAD_WINDOWS = (5, 20, 60)


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


def _build_common(X):
    Xs = X.sort_index()
    prices = Xs.ffill()
    lme_prices = prices[LME_COLS]
    lme_logp = np.log(lme_prices.clip(lower=1e-9))
    lme_ret1 = lme_logp.diff()
    lme_block = pd.concat([lme_prices, lme_ret1.add_suffix("_ret1")], axis=1).fillna(0.0).astype(np.float32).values

    jpx_prices = prices[[c for c in JPX_COLS if c in prices.columns]]
    jpx_logp = np.log(jpx_prices.clip(lower=1e-9))
    jpx_ret1 = jpx_logp.diff()
    jpx_block = pd.concat([jpx_prices, jpx_ret1.add_suffix("_ret1")], axis=1).fillna(0.0).astype(np.float32).values

    date_id = Xs.index.to_numpy()
    cal = np.stack([(date_id % 5).astype(np.float32), (date_id % 22).astype(np.float32),
                    (date_id / 2000.0).astype(np.float32)], axis=1)

    common = np.hstack([lme_block, jpx_block, cal])
    return Xs.index, common


def _build_spread_cache(X, target_pairs):
    Xs = X.sort_index(); prices = Xs.ffill()
    pairs = set()
    for p in target_pairs["pair"]:
        if " - " in p:
            a, b = [s.strip() for s in p.split(" - ")]; pairs.add((a, b))
    cache = {}
    for a, b in pairs:
        if a in prices.columns and b in prices.columns:
            lpa = np.log(prices[a].clip(lower=1e-12).values.astype(np.float64))
            lpb = np.log(prices[b].clip(lower=1e-12).values.astype(np.float64))
            cache[(a, b)] = _spread_multi(lpa, lpb)
    return cache


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
        if " - " in p: return tuple(s.strip() for s in p.split(" - "))
        return None
    def fit(self, X, y):
        self._target_cols = list(y.columns); self._X_train = X.copy()
        full_X = self._combine(None)
        idx, common = _build_common(full_X)
        spread_cache = _build_spread_cache(full_X, self._pairs_df)
        pos = idx.get_indexer(y.index)
        common = common[pos]
        spread_cache = {k: v[pos] for k, v in spread_cache.items()}
        params = dict(device="cuda", tree_method="hist", n_estimators=100,
                      max_depth=4, learning_rate=0.05, verbosity=0, n_jobs=1)
        jobs = []
        for t in self._target_cols:
            pair = self._pair_tuple(t)
            X_j = np.hstack([common, spread_cache[pair]]) if pair in spread_cache else common
            jobs.append((X_j, y[t].values, params))
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
            self._models = list(ex.map(_fit_one, jobs))
    def predict(self, X):
        full_X = self._combine(X)
        idx, common = _build_common(full_X)
        spread_cache = _build_spread_cache(full_X, self._pairs_df)
        pos = idx.get_indexer(X.index)
        common = common[pos]
        spread_cache = {k: v[pos] for k, v in spread_cache.items()}
        jobs = []
        for j, t in enumerate(self._target_cols):
            pair = self._pair_tuple(t)
            X_j = np.hstack([common, spread_cache[pair]]) if pair in spread_cache else common
            jobs.append((X_j, self._models[j]))
        out = np.zeros((len(X), len(self._target_cols)), dtype=np.float32)
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
            results = list(ex.map(_predict_one, jobs))
        for j, p in enumerate(results):
            if p is not None: out[:, j] = p
        return pd.DataFrame(out, index=X.index, columns=self._target_cols)
