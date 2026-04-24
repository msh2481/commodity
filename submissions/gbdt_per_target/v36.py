"""v36: v32 ensemble + K=200 selection trick with INTERNAL 3-fold CV.

Inside fit():
  1. Chronologically split training data into 3 internal folds.
  2. For each, fit v15-style model on the other 2 folds and predict this one.
     (v15 alone used for ranking, not full 4-way ensemble, to save compute.)
  3. Per-target Pearson(OOF pred, truth) on training → rank → top K=200.
  4. Fit full v32 ensemble on all training data.

Inside predict():
  Run v32 ensemble, keep top-K predictions, fill others with per-day median of K.
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
TOP_K = 200
N_INTERNAL_FOLDS = 3

LME_COLS = ["LME_AH_Close", "LME_CA_Close", "LME_PB_Close", "LME_ZS_Close"]
JPX_MAIN = ["JPX_Gold_Standard_Futures_Close", "JPX_Platinum_Standard_Futures_Close"]
SPREAD_WINDOWS_MULTI = (5, 20, 60)
ENSEMBLE_WEIGHTS = (2, 2, 3, 1)  # v15, v17, v28, v1


def _spread_5(logp_a, logp_b):
    spread = (logp_a - logp_b).astype(np.float32)
    s = pd.Series(spread)
    sma = s.rolling(5, min_periods=3).mean().values.astype(np.float32)
    std = s.rolling(5, min_periods=3).std().values.astype(np.float32)
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
    s5 = {}; smulti = {}
    for a, b in pairs:
        if a in prices.columns and b in prices.columns:
            lpa = np.log(prices[a].clip(lower=1e-12).values.astype(np.float64))
            lpb = np.log(prices[b].clip(lower=1e-12).values.astype(np.float64))
            s5[(a, b)] = _spread_5(lpa, lpb)
            smulti[(a, b)] = _spread_multi(lpa, lpb)
    return s5, smulti


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


def _predict_quad(args):
    X15, m15, X17, m17, X28, m28, X1, m1 = args
    preds = []; ws = []
    if m15 is not None: preds.append(m15.predict(X15)); ws.append(ENSEMBLE_WEIGHTS[0])
    if m17 is not None: preds.append(m17.predict(X17)); ws.append(ENSEMBLE_WEIGHTS[1])
    if m28 is not None: preds.append(m28.predict(X28)); ws.append(ENSEMBLE_WEIGHTS[2])
    if m1 is not None: preds.append(m1.predict(X1)); ws.append(ENSEMBLE_WEIGHTS[3])
    if not preds: return None
    w = np.array(ws) / sum(ws)
    return np.sum([w[i] * p for i, p in enumerate(preds)], axis=0)


class Model:
    device = "cuda"

    def __init__(self):
        self._target_cols = None
        self._m15 = []; self._m17 = []; self._m28 = []; self._m1 = []
        self._pairs_df = pd.read_csv(TARGET_PAIRS_CSV)
        self._pair_of = dict(zip(self._pairs_df["target"], self._pairs_df["pair"]))
        self._X_train = None
        self._top_k_idx: np.ndarray | None = None

    def _combine(self, test_X=None):
        if test_X is None: return self._X_train.sort_index()
        c = pd.concat([self._X_train, test_X])
        return c[~c.index.duplicated(keep="first")].sort_index()

    def _pair_tuple(self, t):
        p = self._pair_of[t]
        if " - " in p: return tuple(s.strip() for s in p.split(" - "))
        return None

    def _fit_v15_only(self, X_full_sorted, y, idx, lme, cal, smulti, params):
        """Fit v15-style models (one per target) and return the fitted list."""
        pos = idx.get_indexer(y.index)
        lme_p = lme[pos]; cal_p = cal[pos]
        sm_p = {k: v[pos] for k, v in smulti.items()}
        common = np.hstack([lme_p, cal_p])
        jobs = []
        for tgt in self._target_cols:
            pair = self._pair_tuple(tgt)
            X_j = np.hstack([common, sm_p[pair]]) if pair in sm_p else common
            jobs.append((X_j, y[tgt].values, params))
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
            return list(ex.map(_fit_one, jobs))

    def _predict_v15_only(self, X_full_sorted, test_idx, lme, cal, smulti, models):
        """Predict v15-style on test rows of X."""
        pos = X_full_sorted.index.get_indexer(test_idx)
        lme_p = lme[pos]; cal_p = cal[pos]
        sm_p = {k: v[pos] for k, v in smulti.items()}
        common = np.hstack([lme_p, cal_p])
        jobs = []
        for j, tgt in enumerate(self._target_cols):
            pair = self._pair_tuple(tgt)
            X_j = np.hstack([common, sm_p[pair]]) if pair in sm_p else common
            jobs.append((X_j, models[j]))
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
            return list(ex.map(_predict_one, jobs))

    def _run_internal_cv_for_K(self, y, full_X_sorted, idx, lme, cal, smulti, params):
        """3-fold chronological internal CV using v15 to get OOF preds."""
        date_ids = y.index.to_numpy()
        date_ids_sorted = np.sort(date_ids)
        splits = np.array_split(date_ids_sorted, N_INTERNAL_FOLDS)
        T = len(self._target_cols)
        D = len(date_ids_sorted)
        oof_preds = np.full((D, T), np.nan, dtype=np.float32)
        pos_map = {d: i for i, d in enumerate(date_ids_sorted)}
        for s_i, test_dates in enumerate(splits):
            tr_dates = np.setdiff1d(date_ids_sorted, test_dates, assume_unique=True)
            y_tr = y.loc[tr_dates]
            models = self._fit_v15_only(full_X_sorted, y_tr, idx, lme, cal, smulti, params)
            preds_list = self._predict_v15_only(full_X_sorted, test_dates, lme, cal, smulti, models)
            test_pos = [pos_map[d] for d in test_dates]
            for j, p in enumerate(preds_list):
                if p is not None:
                    oof_preds[test_pos, j] = p
        Y_sorted = y.loc[date_ids_sorted].values
        corrs = np.full(T, -999.0)
        for j in range(T):
            p = oof_preds[:, j]; t = Y_sorted[:, j]
            m = ~(np.isnan(p) | np.isnan(t))
            if m.sum() > 50 and p[m].std() > 1e-12 and t[m].std() > 1e-12:
                corrs[j] = np.corrcoef(p[m], t[m])[0, 1]
        return np.argsort(-corrs)[:TOP_K]

    def fit(self, X, y):
        self._target_cols = list(y.columns); self._X_train = X.copy()
        full_X = self._combine(None)
        idx, lme = _build_lme(full_X)
        jpx = _build_jpx(full_X)
        cal = _calendar(idx)
        s5, smulti = _build_spreads_all(full_X, self._pairs_df)

        params = dict(device="cuda", tree_method="hist", n_estimators=100,
                      max_depth=4, learning_rate=0.05, verbosity=0, n_jobs=1)

        # Internal CV to pick top-K
        self._top_k_idx = self._run_internal_cv_for_K(y, full_X, idx, lme, cal, smulti, params)

        # Fit full v32 ensemble on all training data
        pos = idx.get_indexer(y.index)
        lme_p = lme[pos]; jpx_p = jpx[pos]; cal_p = cal[pos]
        s5_p = {k: v[pos] for k, v in s5.items()}
        sm_p = {k: v[pos] for k, v in smulti.items()}
        common_v15 = np.hstack([lme_p, cal_p])
        common_v17 = lme_p
        common_v28 = np.hstack([lme_p, jpx_p, cal_p])
        common_v1 = lme_p

        j15, j17, j28, j1 = [], [], [], []
        for tgt in self._target_cols:
            pair = self._pair_tuple(tgt)
            X15 = np.hstack([common_v15, sm_p[pair]]) if pair in sm_p else common_v15
            X17 = np.hstack([common_v17, s5_p[pair]]) if pair in s5_p else common_v17
            X28 = np.hstack([common_v28, sm_p[pair]]) if pair in sm_p else common_v28
            X1  = common_v1
            j15.append((X15, y[tgt].values, params))
            j17.append((X17, y[tgt].values, params))
            j28.append((X28, y[tgt].values, params))
            j1.append((X1, y[tgt].values, params))
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
            self._m15 = list(ex.map(_fit_one, j15))
            self._m17 = list(ex.map(_fit_one, j17))
            self._m28 = list(ex.map(_fit_one, j28))
            self._m1 = list(ex.map(_fit_one, j1))

    def predict(self, X):
        full_X = self._combine(X)
        idx, lme = _build_lme(full_X)
        jpx = _build_jpx(full_X)
        cal = _calendar(idx)
        s5, smulti = _build_spreads_all(full_X, self._pairs_df)
        pos = idx.get_indexer(X.index)
        lme_p = lme[pos]; jpx_p = jpx[pos]; cal_p = cal[pos]
        s5_p = {k: v[pos] for k, v in s5.items()}
        sm_p = {k: v[pos] for k, v in smulti.items()}
        common_v15 = np.hstack([lme_p, cal_p])
        common_v17 = lme_p
        common_v28 = np.hstack([lme_p, jpx_p, cal_p])
        common_v1 = lme_p

        jobs = []
        for j, tgt in enumerate(self._target_cols):
            pair = self._pair_tuple(tgt)
            X15 = np.hstack([common_v15, sm_p[pair]]) if pair in sm_p else common_v15
            X17 = np.hstack([common_v17, s5_p[pair]]) if pair in s5_p else common_v17
            X28 = np.hstack([common_v28, sm_p[pair]]) if pair in sm_p else common_v28
            X1  = common_v1
            jobs.append((X15, self._m15[j], X17, self._m17[j],
                         X28, self._m28[j], X1, self._m1[j]))
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
            results = list(ex.map(_predict_quad, jobs))

        D = len(X); T = len(self._target_cols)
        raw = np.zeros((D, T), dtype=np.float32)
        for j, p in enumerate(results):
            if p is not None: raw[:, j] = p

        kept = raw[:, self._top_k_idx]
        med = np.nanmedian(kept, axis=1, keepdims=True)
        out = np.broadcast_to(med, raw.shape).copy()
        out[:, self._top_k_idx] = raw[:, self._top_k_idx]
        return pd.DataFrame(out, index=X.index, columns=self._target_cols)
