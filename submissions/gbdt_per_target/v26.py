"""v25: ensemble of 3 feature variants (v14, v15, v17) averaged per target.

v14: 8 LME + 3 spread(20d) = 11 features
v15: 8 LME + 3 cal + 7 spread(5,20,60) = 18 features
v17: 8 LME + 3 spread(5d) = 11 features

All use v1 hparams (100 trees depth 4 lr 0.05). Thread-parallel.
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
SPREAD_WINDOWS_MULTI = (5, 20, 60)


def _spread_window(logp_a, logp_b, w):
    spread = (logp_a - logp_b).astype(np.float32)
    s = pd.Series(spread)
    sma = s.rolling(w, min_periods=max(3, w // 2)).mean().values.astype(np.float32)
    std = s.rolling(w, min_periods=max(3, w // 2)).std().values.astype(np.float32)
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


def _build_spreads_all(X, target_pairs):
    Xs = X.sort_index(); prices = Xs.ffill()
    pairs = set()
    for p in target_pairs["pair"]:
        if " - " in p:
            a, b = [s.strip() for s in p.split(" - ")]; pairs.add((a, b))
    s20 = {}; s5 = {}; smulti = {}
    for a, b in pairs:
        if a in prices.columns and b in prices.columns:
            lpa = np.log(prices[a].clip(lower=1e-12).values.astype(np.float64))
            lpb = np.log(prices[b].clip(lower=1e-12).values.astype(np.float64))
            s20[(a, b)] = _spread_window(lpa, lpb, 20)
            s5[(a, b)] = _spread_window(lpa, lpb, 5)
            smulti[(a, b)] = _spread_multi(lpa, lpb)
    return s20, s5, smulti


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


def _predict_triple(args):  # weighted 1-2-1
    X14, m14, X15, m15, X17, m17 = args
    preds = []
    if m14 is not None: preds.append(m14.predict(X14))
    if m15 is not None: preds.append(m15.predict(X15))
    if m17 is not None: preds.append(m17.predict(X17))
    weights = [1, 2, 1][:len(preds)]; w = np.array(weights)/sum(weights); return np.sum([w[i]*p for i,p in enumerate(preds)], axis=0) if preds else None


class Model:
    device = "cuda"
    def __init__(self):
        self._target_cols = None
        self._models_v14 = []; self._models_v15 = []; self._models_v17 = []
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
        s20, s5, smulti = _build_spreads_all(full_X, self._pairs_df)
        pos = idx.get_indexer(y.index)
        lme = lme[pos]; cal = cal[pos]
        s20 = {k: v[pos] for k, v in s20.items()}
        s5 = {k: v[pos] for k, v in s5.items()}
        smulti = {k: v[pos] for k, v in smulti.items()}
        common_v14_17 = lme  # v14 and v17 share common tail = just LME
        common_v15 = np.hstack([lme, cal])  # v15 has cal

        params = dict(device="cuda", tree_method="hist", n_estimators=100,
                      max_depth=4, learning_rate=0.05, verbosity=0, n_jobs=1)
        jobs14, jobs15, jobs17 = [], [], []
        for tgt in self._target_cols:
            pair = self._pair_tuple(tgt)
            X14 = np.hstack([common_v14_17, s20[pair]]) if pair in s20 else common_v14_17
            X15 = np.hstack([common_v15, smulti[pair]]) if pair in smulti else common_v15
            X17 = np.hstack([common_v14_17, s5[pair]]) if pair in s5 else common_v14_17
            jobs14.append((X14, y[tgt].values, params))
            jobs15.append((X15, y[tgt].values, params))
            jobs17.append((X17, y[tgt].values, params))
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
            self._models_v14 = list(ex.map(_fit_one, jobs14))
            self._models_v15 = list(ex.map(_fit_one, jobs15))
            self._models_v17 = list(ex.map(_fit_one, jobs17))
    def predict(self, X):
        full_X = self._combine(X)
        idx, lme = _build_lme(full_X); cal = _calendar(idx)
        s20, s5, smulti = _build_spreads_all(full_X, self._pairs_df)
        pos = idx.get_indexer(X.index)
        lme = lme[pos]; cal = cal[pos]
        s20 = {k: v[pos] for k, v in s20.items()}
        s5 = {k: v[pos] for k, v in s5.items()}
        smulti = {k: v[pos] for k, v in smulti.items()}
        common_v14_17 = lme
        common_v15 = np.hstack([lme, cal])

        jobs = []
        for j, tgt in enumerate(self._target_cols):
            pair = self._pair_tuple(tgt)
            X14 = np.hstack([common_v14_17, s20[pair]]) if pair in s20 else common_v14_17
            X15 = np.hstack([common_v15, smulti[pair]]) if pair in smulti else common_v15
            X17 = np.hstack([common_v14_17, s5[pair]]) if pair in s5 else common_v14_17
            jobs.append((X14, self._models_v14[j],
                         X15, self._models_v15[j],
                         X17, self._models_v17[j]))
        out = np.zeros((len(X), len(self._target_cols)), dtype=np.float32)
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
            results = list(ex.map(_predict_triple, jobs))
        for j, p in enumerate(results):
            if p is not None: out[:, j] = p
        return pd.DataFrame(out, index=X.index, columns=self._target_cols)
