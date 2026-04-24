"""v29: weighted ensemble v14(1) + v15(2) + v28(2) per target.

v14: 8 LME + 3 spread(20d) = 11
v15: 8 LME + 3 cal + 7 spread(5,20,60) = 18
v28: 8 LME + 3 cal + 6 JPX (Gold/Platinum Std) + 7 spread(5,20,60) = 24

Weights 1:2:2 (v15 and v28 get more weight).
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
JPX_MAIN = ["JPX_Gold_Standard_Futures_Close", "JPX_Platinum_Standard_Futures_Close"]
SPREAD_WINDOWS_MULTI = (5, 20, 60)


def _spread_single_20(logp_a, logp_b):
    spread = (logp_a - logp_b).astype(np.float32)
    s = pd.Series(spread)
    sma = s.rolling(20, min_periods=10).mean().values.astype(np.float32)
    std = s.rolling(20, min_periods=10).std().values.astype(np.float32)
    dev = (spread - sma).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(std > 1e-9, dev / std, np.nan).astype(np.float32)
    return np.stack([spread, dev, z], axis=1)


def _spread_multi(logp_a, logp_b, windows=SPREAD_WINDOWS_MULTI):
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


def _build_jpx(X):
    Xs = X.sort_index(); prices = Xs.ffill()
    cols = [c for c in JPX_MAIN if c in prices.columns]
    if not cols:
        return np.empty((len(Xs), 0), dtype=np.float32)
    j = prices[cols]
    j_logp = np.log(j.clip(lower=1e-9))
    j_ret1 = j_logp.diff()
    j_vol20 = j_ret1.rolling(20, min_periods=10).std()
    return pd.concat([j, j_ret1.add_suffix("_ret1"), j_vol20.add_suffix("_vol20")],
                     axis=1).fillna(0.0).astype(np.float32).values


def _calendar(index):
    d = index.to_numpy()
    return np.stack([(d % 5).astype(np.float32), (d % 22).astype(np.float32),
                     (d / 2000.0).astype(np.float32)], axis=1)


def _build_spreads_all(X, target_pairs):
    Xs = X.sort_index(); prices = Xs.ffill()
    pairs = set()
    for p in target_pairs["pair"]:
        if " - " in p:
            a, b = [s.strip() for s in p.split(" - ")]; pairs.add((a, b))
    s20 = {}; smulti = {}
    for a, b in pairs:
        if a in prices.columns and b in prices.columns:
            lpa = np.log(prices[a].clip(lower=1e-12).values.astype(np.float64))
            lpb = np.log(prices[b].clip(lower=1e-12).values.astype(np.float64))
            s20[(a, b)] = _spread_single_20(lpa, lpb)
            smulti[(a, b)] = _spread_multi(lpa, lpb)
    return s20, smulti


def _fit_one(args):
    X_np, yt, params = args
    mask = ~np.isnan(yt)
    if mask.sum() < 50: return None
    m = xgb.XGBRegressor(**params); m.fit(X_np[mask], yt[mask].astype(np.float32))
    return m


def _predict_weighted(args):
    X14, m14, X15, m15, X28, m28, weights = args
    preds = []; ws = []
    if m14 is not None: preds.append(m14.predict(X14)); ws.append(weights[0])
    if m15 is not None: preds.append(m15.predict(X15)); ws.append(weights[1])
    if m28 is not None: preds.append(m28.predict(X28)); ws.append(weights[2])
    if not preds: return None
    w = np.array(ws) / sum(ws)
    return np.sum([w[i] * p for i, p in enumerate(preds)], axis=0)


class Model:
    device = "cuda"
    def __init__(self):
        self._target_cols = None
        self._models_v14 = []; self._models_v15 = []; self._models_v28 = []
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
        idx, lme = _build_lme(full_X)
        jpx = _build_jpx(full_X)
        cal = _calendar(idx)
        s20, smulti = _build_spreads_all(full_X, self._pairs_df)
        pos = idx.get_indexer(y.index)
        lme = lme[pos]; jpx = jpx[pos]; cal = cal[pos]
        s20 = {k: v[pos] for k, v in s20.items()}
        smulti = {k: v[pos] for k, v in smulti.items()}
        common_v14 = lme
        common_v15 = np.hstack([lme, cal])
        common_v28 = np.hstack([lme, jpx, cal])

        params = dict(device="cuda", tree_method="hist", n_estimators=100,
                      max_depth=4, learning_rate=0.05, verbosity=0, n_jobs=1)
        j14, j15, j28 = [], [], []
        for tgt in self._target_cols:
            pair = self._pair_tuple(tgt)
            X14 = np.hstack([common_v14, s20[pair]]) if pair in s20 else common_v14
            X15 = np.hstack([common_v15, smulti[pair]]) if pair in smulti else common_v15
            X28 = np.hstack([common_v28, smulti[pair]]) if pair in smulti else common_v28
            j14.append((X14, y[tgt].values, params))
            j15.append((X15, y[tgt].values, params))
            j28.append((X28, y[tgt].values, params))
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
            self._models_v14 = list(ex.map(_fit_one, j14))
            self._models_v15 = list(ex.map(_fit_one, j15))
            self._models_v28 = list(ex.map(_fit_one, j28))
    def predict(self, X):
        full_X = self._combine(X)
        idx, lme = _build_lme(full_X)
        jpx = _build_jpx(full_X)
        cal = _calendar(idx)
        s20, smulti = _build_spreads_all(full_X, self._pairs_df)
        pos = idx.get_indexer(X.index)
        lme = lme[pos]; jpx = jpx[pos]; cal = cal[pos]
        s20 = {k: v[pos] for k, v in s20.items()}
        smulti = {k: v[pos] for k, v in smulti.items()}
        common_v14 = lme
        common_v15 = np.hstack([lme, cal])
        common_v28 = np.hstack([lme, jpx, cal])

        weights = (1, 2, 2)
        jobs = []
        for j, tgt in enumerate(self._target_cols):
            pair = self._pair_tuple(tgt)
            X14 = np.hstack([common_v14, s20[pair]]) if pair in s20 else common_v14
            X15 = np.hstack([common_v15, smulti[pair]]) if pair in smulti else common_v15
            X28 = np.hstack([common_v28, smulti[pair]]) if pair in smulti else common_v28
            jobs.append((X14, self._models_v14[j],
                         X15, self._models_v15[j],
                         X28, self._models_v28[j],
                         weights))
        out = np.zeros((len(X), len(self._target_cols)), dtype=np.float32)
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
            results = list(ex.map(_predict_weighted, jobs))
        for j, p in enumerate(results):
            if p is not None: out[:, j] = p
        return pd.DataFrame(out, index=X.index, columns=self._target_cols)
