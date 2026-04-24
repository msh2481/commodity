"""v38_tune: same pipeline as v38 but loads XGBoost hparams from _tune/hparams.json.

This file is used by tune.py to run skopt-driven hparam search. Each trial
rewrites _tune/hparams.json before calling ./validate, and this module
re-reads it at Model() instantiation time (which happens fresh per fold).
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


TARGET_PAIRS_CSV = "/root/commodity/data/target_pairs.csv"
HPARAMS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tune", "hparams.json")
N_THREADS_FIT = 8

LME_COLS = ["LME_AH_Close", "LME_CA_Close", "LME_PB_Close", "LME_ZS_Close"]
JPX_MAIN = ["JPX_Gold_Standard_Futures_Close", "JPX_Platinum_Standard_Futures_Close"]
SPREAD_WINDOWS = (5, 20, 60)


def _diff_n(a, k):
    n = len(a); out = np.full(n, np.nan, dtype=np.float32)
    if k < n: out[k:] = (a[k:] - a[:-k]).astype(np.float32)
    return out


def _build_common(X):
    Xs = X.sort_index()
    prices = Xs.ffill()
    lme = prices[LME_COLS]
    lme_logp = np.log(lme.clip(lower=1e-9))
    lme_ret1 = lme_logp.diff()
    lme_block = pd.concat([
        lme,
        lme_ret1.add_suffix("_ret1"),
    ], axis=1).fillna(0.0).astype(np.float32).values

    jpx_cols = [c for c in JPX_MAIN if c in prices.columns]
    if jpx_cols:
        jpx = prices[jpx_cols]
        jpx_logp = np.log(jpx.clip(lower=1e-9))
        jpx_ret1 = jpx_logp.diff()
        jpx_vol20 = jpx_ret1.rolling(20, min_periods=10).std()
        jpx_block = pd.concat([jpx, jpx_ret1.add_suffix("_ret1"), jpx_vol20.add_suffix("_vol20")],
                              axis=1).fillna(0.0).astype(np.float32).values
    else:
        jpx_block = np.empty((len(Xs), 0), dtype=np.float32)

    d = Xs.index.to_numpy()
    cal = np.stack([(d % 5).astype(np.float32),
                    (d % 22).astype(np.float32),
                    (d / 2000.0).astype(np.float32)], axis=1)

    return Xs.index, np.hstack([lme_block, jpx_block, cal])


def _spread_features(logp_a, logp_b, lag_h):
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
    sf = np.asarray(spread, dtype=np.float64)
    feats.append(_diff_n(sf, lag_h))
    feats.append(_diff_n(sf, 2 * lag_h))
    spread_ret1 = _diff_n(sf, 1)
    spread_vol20 = pd.Series(spread_ret1).rolling(20, min_periods=10).std().values.astype(np.float32)
    feats.append(spread_vol20)
    return np.stack(feats, axis=1)


def _asset_min(logp):
    n = len(logp)
    ret5 = np.full(n, np.nan, dtype=np.float32)
    if 5 < n: ret5[5:] = (logp[5:] - logp[:-5]).astype(np.float32)
    ret1 = np.full(n, np.nan, dtype=np.float32)
    if 1 < n: ret1[1:] = (logp[1:] - logp[:-1]).astype(np.float32)
    vol20 = pd.Series(ret1).rolling(20, min_periods=10).std().values.astype(np.float32)
    return np.stack([ret5, vol20], axis=1)


def _build_caches(X, target_pairs):
    Xs = X.sort_index(); prices = Xs.ffill()
    assets, pair_lag = set(), set()
    for _, row in target_pairs.iterrows():
        if " - " in row["pair"]:
            a, b = [s.strip() for s in row["pair"].split(" - ")]
            assets.update([a, b]); pair_lag.add((a, b, int(row["lag"])))
        else:
            assets.add(row["pair"].strip())

    asset_cache = {}
    for a in assets:
        if a in prices.columns:
            lp = np.log(prices[a].clip(lower=1e-12).values.astype(np.float64))
            asset_cache[a] = _asset_min(lp)

    spread_cache = {}
    for a, b, lag in pair_lag:
        if a in prices.columns and b in prices.columns:
            lpa = np.log(prices[a].clip(lower=1e-12).values.astype(np.float64))
            lpb = np.log(prices[b].clip(lower=1e-12).values.astype(np.float64))
            spread_cache[(a, b, lag)] = _spread_features(lpa, lpb, lag)

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


def _load_hparams():
    with open(HPARAMS_JSON) as f:
        user = json.load(f)
    fixed = dict(device="cuda", tree_method="hist", verbosity=0, n_jobs=1)
    return {**fixed, **user}


class Model:
    device = "cuda"

    def __init__(self):
        self.HPARAMS = _load_hparams()
        self._target_cols = None
        self._models = []
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

        self._models = [None] * len(self._target_cols)
        jobs = []
        idx_for_jobs = []
        for j in self._keep_idx:
            t = self._target_cols[j]
            X_j = self._features_for_target(t, common, asset_cache, spread_cache)
            jobs.append((X_j, y[t].values, self.HPARAMS))
            idx_for_jobs.append(j)

        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
            results = list(ex.map(_fit_one, jobs))
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

        jobs = []
        idx_for_jobs = []
        for j in self._keep_idx:
            t = self._target_cols[j]
            X_j = self._features_for_target(t, common, asset_cache, spread_cache)
            jobs.append((X_j, self._models[j]))
            idx_for_jobs.append(j)

        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
            results = list(ex.map(_predict_one, jobs))

        D = len(X); T = len(self._target_cols)
        raw = np.full((D, T), np.nan, dtype=np.float32)
        for j, p in zip(idx_for_jobs, results):
            if p is not None: raw[:, j] = p

        kept = raw[:, self._keep_idx]
        med = np.nanmedian(kept, axis=1, keepdims=True)
        out = np.broadcast_to(med, raw.shape).copy()
        out[:, self._keep_idx] = raw[:, self._keep_idx]
        out = np.nan_to_num(out, nan=0.0)
        return pd.DataFrame(out, index=X.index, columns=self._target_cols)
