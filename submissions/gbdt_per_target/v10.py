"""v10: v9 focused features + raw log-prices + more horizons.

Adds on top of v9:
  - raw log-price of A, B, and market proxies (level info; v1 had raw LME prices)
  - extra horizon: ret2, ret3, ret4 (lag-matched to target horizons 1-4)
  - longer vol window (vol60) and shorter (vol5) per asset
  - pair ratio dev/z at 5-day window (short-term reversion)

Conservative hparams (250 trees depth 4, mcw=10, cols 0.7, lambda 3).
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

MARKET_PROXIES = ["LME_CA_Close", "US_Stock_VT_adj_close", "FX_USDJPY"]


def _diff_n(a: np.ndarray, k: int) -> np.ndarray:
    n = len(a)
    out = np.full(n, np.nan, dtype=np.float32)
    if k < n:
        out[k:] = a[k:] - a[:-k]
    return out


def _asset_feats(logp: np.ndarray) -> np.ndarray:
    ret1 = _diff_n(logp, 1)
    ret2 = _diff_n(logp, 2)
    ret3 = _diff_n(logp, 3)
    ret4 = _diff_n(logp, 4)
    ret5 = _diff_n(logp, 5)
    ret10 = _diff_n(logp, 10)
    ret20 = _diff_n(logp, 20)
    r1s = pd.Series(ret1)
    vol5 = r1s.rolling(5, min_periods=3).std().values.astype(np.float32)
    vol20 = r1s.rolling(20, min_periods=10).std().values.astype(np.float32)
    vol60 = r1s.rolling(60, min_periods=30).std().values.astype(np.float32)
    mean20 = r1s.rolling(20, min_periods=10).mean().values.astype(np.float32)
    # log-price level standardised over 250d
    logp_s = pd.Series(logp.astype(np.float64))
    logp_mu = logp_s.rolling(250, min_periods=60).mean()
    logp_sd = logp_s.rolling(250, min_periods=60).std().replace(0, np.nan)
    logp_z = ((logp_s - logp_mu) / logp_sd).values.astype(np.float32)
    return np.stack(
        [ret1, ret2, ret3, ret4, ret5, ret10, ret20,
         vol5, vol20, vol60, mean20, logp_z, logp.astype(np.float32)],
        axis=1,
    )


def _spread_feats(logp_a: np.ndarray, logp_b: np.ndarray) -> np.ndarray:
    spread = (logp_a - logp_b).astype(np.float32)
    s = pd.Series(spread)
    sma20 = s.rolling(20, min_periods=10).mean().values.astype(np.float32)
    std20 = s.rolling(20, min_periods=10).std().values.astype(np.float32)
    sma5 = s.rolling(5, min_periods=3).mean().values.astype(np.float32)
    std5 = s.rolling(5, min_periods=3).std().values.astype(np.float32)
    dev20 = spread - sma20
    dev5 = spread - sma5
    with np.errstate(divide="ignore", invalid="ignore"):
        z20 = np.where(std20 > 1e-9, dev20 / std20, np.nan).astype(np.float32)
        z5 = np.where(std5 > 1e-9, dev5 / std5, np.nan).astype(np.float32)
    return np.stack([spread, dev20, z20, dev5, z5], axis=1)


def _market_feats(logp: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            _diff_n(logp, 1),
            _diff_n(logp, 5),
            _diff_n(logp, 10),
            _diff_n(logp, 20),
            logp.astype(np.float32),
        ],
        axis=1,
    )


def _calendar_feats(index: pd.Index) -> np.ndarray:
    date_id = index.to_numpy()
    return np.stack(
        [
            (date_id % 5).astype(np.float32),
            (date_id % 22).astype(np.float32),
            (date_id / 2000.0).astype(np.float32),
        ],
        axis=1,
    )


def _prep_caches(X: pd.DataFrame, target_pairs: pd.DataFrame):
    Xs = X.sort_index()
    prices = Xs.ffill()

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
            asset_cache[a] = _asset_feats(logp)

    pair_cache = {}
    for a, b in pairs:
        if a in prices.columns and b in prices.columns:
            lpa = np.log(prices[a].clip(lower=1e-12).values.astype(np.float64))
            lpb = np.log(prices[b].clip(lower=1e-12).values.astype(np.float64))
            pair_cache[(a, b)] = _spread_feats(lpa, lpb)

    mkt_arrays = []
    for mk in MARKET_PROXIES:
        if mk in prices.columns:
            logp = np.log(prices[mk].clip(lower=1e-12).values.astype(np.float64))
            mkt_arrays.append(_market_feats(logp))
    mkt_concat = np.hstack(mkt_arrays) if mkt_arrays else np.empty((len(Xs), 0), dtype=np.float32)
    cal = _calendar_feats(Xs.index)
    common_tail = np.hstack([mkt_concat, cal])

    return Xs.index, asset_cache, pair_cache, common_tail


def _feats_for_target(target_pair: str, asset_cache, pair_cache, common_tail):
    parts = []
    if " - " in target_pair:
        a, b = [s.strip() for s in target_pair.split(" - ")]
        if a in asset_cache:
            parts.append(asset_cache[a])
        if b in asset_cache:
            parts.append(asset_cache[b])
        if (a, b) in pair_cache:
            parts.append(pair_cache[(a, b)])
    else:
        a = target_pair.strip()
        if a in asset_cache:
            parts.append(asset_cache[a])
    parts.append(common_tail)
    return np.hstack(parts)


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
        self._target_cols: list[str] | None = None
        self._models: list[xgb.XGBRegressor | None] = []
        self._pairs_df = pd.read_csv(TARGET_PAIRS_CSV)
        self._pair_of = dict(zip(self._pairs_df["target"], self._pairs_df["pair"]))
        self._X_train: pd.DataFrame | None = None

    def _combine_X(self, test_X: pd.DataFrame | None = None) -> pd.DataFrame:
        if test_X is None:
            return self._X_train.sort_index()
        combined = pd.concat([self._X_train, test_X])
        return combined[~combined.index.duplicated(keep="first")].sort_index()

    def fit(self, X, y):
        self._target_cols = list(y.columns)
        self._X_train = X.copy()
        full_X = self._combine_X(None)
        idx, asset_cache, pair_cache, common_tail = _prep_caches(full_X, self._pairs_df)
        pos = idx.get_indexer(y.index)
        asset_cache = {k: v[pos] for k, v in asset_cache.items()}
        pair_cache = {k: v[pos] for k, v in pair_cache.items()}
        common_tail = common_tail[pos]

        params = dict(
            device="cuda",
            tree_method="hist",
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_lambda=3.0,
            verbosity=0,
            n_jobs=1,
        )
        jobs = []
        for tgt in self._target_cols:
            X_j = _feats_for_target(self._pair_of[tgt], asset_cache, pair_cache, common_tail)
            jobs.append((X_j, y[tgt].values, params))
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
            self._models = list(ex.map(_fit_one, jobs))

    def predict(self, X):
        full_X = self._combine_X(X)
        idx, asset_cache, pair_cache, common_tail = _prep_caches(full_X, self._pairs_df)
        pos = idx.get_indexer(X.index)
        asset_cache = {k: v[pos] for k, v in asset_cache.items()}
        pair_cache = {k: v[pos] for k, v in pair_cache.items()}
        common_tail = common_tail[pos]

        jobs = []
        for j, tgt in enumerate(self._target_cols):
            X_j = _feats_for_target(self._pair_of[tgt], asset_cache, pair_cache, common_tail)
            jobs.append((X_j, self._models[j]))
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
            results = list(ex.map(_predict_one, jobs))
        out = np.zeros((len(X), len(self._target_cols)), dtype=np.float32)
        for j, p in enumerate(results):
            if p is not None:
                out[:, j] = p
        return pd.DataFrame(out, index=X.index, columns=self._target_cols)
