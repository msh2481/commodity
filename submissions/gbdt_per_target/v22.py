"""v22: ensemble of v14-style (single-window spread) and v15-style (multi-window).

Fits both feature sets per target, averages predictions.
v14 features: 8 LME + 3 spread(20d) = 11
v15 features: 8 LME + 3 cal + 7 spread(5,20,60) = 18
Final prediction = (v14_pred + v15_pred) / 2.
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


def _spread_single(logp_a, logp_b, w=20):
    spread = (logp_a - logp_b).astype(np.float32)
    s = pd.Series(spread)
    sma = s.rolling(w, min_periods=max(3, w // 2)).mean().values.astype(np.float32)
    std = s.rolling(w, min_periods=max(3, w // 2)).std().values.astype(np.float32)
    dev = (spread - sma).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(std > 1e-9, dev / std, np.nan).astype(np.float32)
    return np.stack([spread, dev, z], axis=1)


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


def _build_spread_caches(X, target_pairs):
    Xs = X.sort_index(); prices = Xs.ffill()
    pairs = set()
    for p in target_pairs["pair"]:
        if " - " in p:
            a, b = [s.strip() for s in p.split(" - ")]; pairs.add((a, b))
    single = {}; multi = {}
    for a, b in pairs:
        if a in prices.columns and b in prices.columns:
            lpa = np.log(prices[a].clip(lower=1e-12).values.astype(np.float64))
            lpb = np.log(prices[b].clip(lower=1e-12).values.astype(np.float64))
            single[(a, b)] = _spread_single(lpa, lpb)
            multi[(a, b)] = _spread_multi(lpa, lpb)
    return single, multi


def _calendar(index):
    d = index.to_numpy()
    return np.stack([(d % 5).astype(np.float32), (d % 22).astype(np.float32),
                     (d / 2000.0).astype(np.float32)], axis=1)


def _fit_one(args):
    X_np, yt, params = args
    mask = ~np.isnan(yt)
    if mask.sum() < 50: return None
    m = xgb.XGBRegressor(**params); m.fit(X_np[mask], yt[mask].astype(np.float32))
    return m


def _predict_pair(args):
    X14, m14, X15, m15 = args
    if m14 is None and m15 is None: return None
    preds = []
    if m14 is not None: preds.append(m14.predict(X14))
    if m15 is not None: preds.append(m15.predict(X15))
    return np.mean(preds, axis=0)


class Model:
    device = "cuda"
    def __init__(self):
        self._target_cols = None
        self._models_v14 = []; self._models_v15 = []
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
        idx, lme = _build_lme(full_X); cal = _calendar(idx)
        single, multi = _build_spread_caches(full_X, self._pairs_df)
        pos = idx.get_indexer(y.index)
        lme = lme[pos]; cal = cal[pos]
        single = {k: v[pos] for k, v in single.items()}
        multi = {k: v[pos] for k, v in multi.items()}
        common_v14 = lme  # v14 common tail = just LME
        common_v15 = np.hstack([lme, cal])

        params = dict(device="cuda", tree_method="hist", n_estimators=100,
                      max_depth=4, learning_rate=0.05, verbosity=0, n_jobs=1)
        jobs14, jobs15 = [], []
        for tgt in self._target_cols:
            pair = self._pair_tuple(tgt)
            X14 = np.hstack([common_v14, single[pair]]) if pair in single else common_v14
            X15 = np.hstack([common_v15, multi[pair]]) if pair in multi else common_v15
            jobs14.append((X14, y[tgt].values, params))
            jobs15.append((X15, y[tgt].values, params))
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
            self._models_v14 = list(ex.map(_fit_one, jobs14))
            self._models_v15 = list(ex.map(_fit_one, jobs15))
    def predict(self, X):
        full_X = self._combine(X)
        idx, lme = _build_lme(full_X); cal = _calendar(idx)
        single, multi = _build_spread_caches(full_X, self._pairs_df)
        pos = idx.get_indexer(X.index)
        lme = lme[pos]; cal = cal[pos]
        single = {k: v[pos] for k, v in single.items()}
        multi = {k: v[pos] for k, v in multi.items()}
        common_v14 = lme
        common_v15 = np.hstack([lme, cal])
        jobs = []
        for j, tgt in enumerate(self._target_cols):
            pair = self._pair_tuple(tgt)
            X14 = np.hstack([common_v14, single[pair]]) if pair in single else common_v14
            X15 = np.hstack([common_v15, multi[pair]]) if pair in multi else common_v15
            jobs.append((X14, self._models_v14[j], X15, self._models_v15[j]))
        out = np.zeros((len(X), len(self._target_cols)), dtype=np.float32)
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
            results = list(ex.map(_predict_pair, jobs))
        for j, p in enumerate(results):
            if p is not None: out[:, j] = p
        return pd.DataFrame(out, index=X.index, columns=self._target_cols)
