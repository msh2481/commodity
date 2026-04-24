"""v33: v15 + minimal FX majors (USDJPY + EURUSD).

Common tail: 8 LME + 3 cal + FX block (2 raw + 2 ret1 + 2 vol20 = 6) = 17.
Plus per-target multi-window spread. Total 24 for spread targets.

Rationale: many targets involve JPY/USD crosses. FX_USDJPY is a frequent
cross-asset driver. Same minimal-addition pattern that made v28 useful in ensemble.
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
FX_MAIN = ["FX_USDJPY", "FX_EURUSD"]
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
    Xs = X.sort_index(); prices = Xs.ffill()
    lme = prices[LME_COLS]
    lme_logp = np.log(lme.clip(lower=1e-9))
    lme_ret1 = lme_logp.diff()
    lme_block = pd.concat([lme, lme_ret1.add_suffix("_ret1")], axis=1).fillna(0.0).astype(np.float32).values

    fx_cols = [c for c in FX_MAIN if c in prices.columns]
    if fx_cols:
        fx = prices[fx_cols]
        fx_logp = np.log(fx.clip(lower=1e-9))
        fx_ret1 = fx_logp.diff()
        fx_vol20 = fx_ret1.rolling(20, min_periods=10).std()
        fx_block = pd.concat([fx, fx_ret1.add_suffix("_ret1"), fx_vol20.add_suffix("_vol20")],
                             axis=1).fillna(0.0).astype(np.float32).values
    else:
        fx_block = np.empty((len(Xs), 0), dtype=np.float32)

    d = Xs.index.to_numpy()
    cal = np.stack([(d % 5).astype(np.float32), (d % 22).astype(np.float32),
                    (d / 2000.0).astype(np.float32)], axis=1)
    return Xs.index, np.hstack([lme_block, fx_block, cal])


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
        for tgt in self._target_cols:
            pair = self._pair_tuple(tgt)
            X_j = np.hstack([common, spread_cache[pair]]) if pair in spread_cache else common
            jobs.append((X_j, y[tgt].values, params))
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
        for j, tgt in enumerate(self._target_cols):
            pair = self._pair_tuple(tgt)
            X_j = np.hstack([common, spread_cache[pair]]) if pair in spread_cache else common
            jobs.append((X_j, self._models[j]))
        out = np.zeros((len(X), len(self._target_cols)), dtype=np.float32)
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
            results = list(ex.map(_predict_one, jobs))
        for j, p in enumerate(results):
            if p is not None: out[:, j] = p
        return pd.DataFrame(out, index=X.index, columns=self._target_cols)
