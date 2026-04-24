"""v19: bagged v15. 3 random-seed XGBoost per target, average predictions.

Same features and hparams as v15. 3x compute but should reduce variance.
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
SEEDS = [42, 7, 123]

LME_COLS = ["LME_AH_Close", "LME_CA_Close", "LME_PB_Close", "LME_ZS_Close"]
SPREAD_WINDOWS = (5, 20, 60)


def _spread_multi(logp_a, logp_b, windows=SPREAD_WINDOWS):
    spread = (logp_a - logp_b).astype(np.float32)
    s = pd.Series(spread)
    feats = [spread]
    for w in windows:
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


def _calendar(index):
    d = index.to_numpy()
    return np.stack([(d % 5).astype(np.float32), (d % 22).astype(np.float32),
                     (d / 2000.0).astype(np.float32)], axis=1)


def _fit_seeds(args):
    """Fit N seeded XGBoost models for one target. Return list of models."""
    X_np, yt, base_params, seeds = args
    mask = ~np.isnan(yt)
    if mask.sum() < 50: return None
    Xm = X_np[mask]; ym = yt[mask].astype(np.float32)
    models = []
    for seed in seeds:
        params = dict(base_params); params["random_state"] = seed
        m = xgb.XGBRegressor(**params); m.fit(Xm, ym)
        models.append(m)
    return models


def _predict_seeds(args):
    X_np, models = args
    if models is None: return None
    return np.mean([m.predict(X_np) for m in models], axis=0)


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
        idx, lme = _build_lme(full_X); cal = _calendar(idx)
        common_tail = np.hstack([lme, cal])
        spread_cache = _build_spread_cache(full_X, self._pairs_df)
        pos = idx.get_indexer(y.index)
        common_tail = common_tail[pos]
        spread_cache = {k: v[pos] for k, v in spread_cache.items()}
        base_params = dict(device="cuda", tree_method="hist", n_estimators=100,
                           max_depth=4, learning_rate=0.05, verbosity=0, n_jobs=1,
                           subsample=0.8, colsample_bytree=0.8)
        jobs = []
        for tgt in self._target_cols:
            pair = self._pair_tuple(tgt)
            X_j = np.hstack([common_tail, spread_cache[pair]]) if pair in spread_cache else common_tail
            jobs.append((X_j, y[tgt].values, base_params, SEEDS))
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
            self._models = list(ex.map(_fit_seeds, jobs))
    def predict(self, X):
        full_X = self._combine(X)
        idx, lme = _build_lme(full_X); cal = _calendar(idx)
        common_tail = np.hstack([lme, cal])
        spread_cache = _build_spread_cache(full_X, self._pairs_df)
        pos = idx.get_indexer(X.index)
        common_tail = common_tail[pos]
        spread_cache = {k: v[pos] for k, v in spread_cache.items()}
        jobs = []
        for j, tgt in enumerate(self._target_cols):
            pair = self._pair_tuple(tgt)
            X_j = np.hstack([common_tail, spread_cache[pair]]) if pair in spread_cache else common_tail
            jobs.append((X_j, self._models[j]))
        out = np.zeros((len(X), len(self._target_cols)), dtype=np.float32)
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
            results = list(ex.map(_predict_seeds, jobs))
        for j, p in enumerate(results):
            if p is not None: out[:, j] = p
        return pd.DataFrame(out, index=X.index, columns=self._target_cols)
