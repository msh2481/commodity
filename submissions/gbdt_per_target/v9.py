"""v9: per-target focused features. Hypothesis: v2's 861 features overfit
because most are noise for any given target. Each target only depends on
2 specific assets (legs of the spread). Use only those + a few market proxies.

For target with pair 'A - B' (or 'A' for single-asset):
  - Asset A features: log_ret_1, log_ret_5, log_ret_20, vol_20, mean_20 (5)
  - Asset B features: same (5, if B exists)
  - Pair spread features: log(A/B), dev_sma_20, zscore_20 (3, if B exists)
  - Market proxies: LME_CA (copper, industrial), US_Stock_VT (global eq),
    FX_USDJPY — each: log_ret_1, log_ret_5, log_ret_20 (9)
  - Calendar: cal_dow, cal_wom, cal_trend (3)

Total ~25 features per target for spreads, ~17 for singles.
Asset & pair features are cached so each is computed only once.

Conservative hparams (200 trees depth 4, mcw=10, cols 0.8, lambda 2).
Thread-parallel fits.
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


def _asset_feats(logp: np.ndarray) -> np.ndarray:
    """5 features: ret1, ret5, ret20, vol20, mean20. Input: (N,) log-prices."""
    # diffs
    n = len(logp)
    def diff_n(a: np.ndarray, k: int) -> np.ndarray:
        out = np.full(n, np.nan, dtype=np.float32)
        if k < n:
            out[k:] = a[k:] - a[:-k]
        return out

    ret1 = diff_n(logp, 1)
    ret5 = diff_n(logp, 5)
    ret20 = diff_n(logp, 20)

    # rolling std and mean of ret1 over window 20
    ret1_s = pd.Series(ret1)
    vol20 = ret1_s.rolling(20, min_periods=10).std().values.astype(np.float32)
    mean20 = ret1_s.rolling(20, min_periods=10).mean().values.astype(np.float32)

    return np.stack([ret1, ret5, ret20, vol20, mean20], axis=1).astype(np.float32)


def _spread_feats(logp_a: np.ndarray, logp_b: np.ndarray) -> np.ndarray:
    spread = (logp_a - logp_b).astype(np.float32)
    spread_s = pd.Series(spread)
    sma = spread_s.rolling(20, min_periods=10).mean().values.astype(np.float32)
    std = spread_s.rolling(20, min_periods=10).std().values.astype(np.float32)
    dev = (spread - sma).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(std > 1e-9, dev / std, np.nan).astype(np.float32)
    return np.stack([spread, dev, z], axis=1)


def _market_feats(logp: np.ndarray) -> np.ndarray:
    n = len(logp)
    def diff_n(a, k):
        out = np.full(n, np.nan, dtype=np.float32)
        if k < n:
            out[k:] = a[k:] - a[:-k]
        return out
    return np.stack([diff_n(logp, 1), diff_n(logp, 5), diff_n(logp, 20)], axis=1).astype(np.float32)


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
    """Precompute per-asset, per-pair, and market feature arrays aligned to X.index."""
    Xs = X.sort_index()
    prices = Xs.ffill()

    assets = set()
    pairs = set()
    for p in target_pairs["pair"]:
        if " - " in p:
            a, b = [s.strip() for s in p.split(" - ")]
            assets.add(a)
            assets.add(b)
            pairs.add((a, b))
        else:
            assets.add(p.strip())

    asset_cache: dict[str, np.ndarray] = {}
    for a in assets:
        if a in prices.columns:
            logp = np.log(prices[a].clip(lower=1e-12).values.astype(np.float64))
            asset_cache[a] = _asset_feats(logp)

    pair_cache: dict[tuple[str, str], np.ndarray] = {}
    for a, b in pairs:
        if a in prices.columns and b in prices.columns:
            logp_a = np.log(prices[a].clip(lower=1e-12).values.astype(np.float64))
            logp_b = np.log(prices[b].clip(lower=1e-12).values.astype(np.float64))
            pair_cache[(a, b)] = _spread_feats(logp_a, logp_b)

    mkt_cache: dict[str, np.ndarray] = {}
    for mk in MARKET_PROXIES:
        if mk in prices.columns:
            logp = np.log(prices[mk].clip(lower=1e-12).values.astype(np.float64))
            mkt_cache[mk] = _market_feats(logp)

    mkt_concat = np.hstack([mkt_cache[mk] for mk in MARKET_PROXIES if mk in mkt_cache])
    cal = _calendar_feats(Xs.index)
    common_tail = np.hstack([mkt_concat, cal]) if mkt_concat.size else cal

    return Xs.index, asset_cache, pair_cache, common_tail


def _feats_for_target(target_pair: str, asset_cache, pair_cache, common_tail):
    parts = []
    if " - " in target_pair:
        a, b = [s.strip() for s in target_pair.split(" - ")]
        if a in asset_cache:
            parts.append(asset_cache[a])
        if b in asset_cache:
            parts.append(asset_cache[b])
        pkey = (a, b)
        if pkey in pair_cache:
            parts.append(pair_cache[pkey])
    else:
        a = target_pair.strip()
        if a in asset_cache:
            parts.append(asset_cache[a])
    parts.append(common_tail)
    return np.hstack(parts) if len(parts) > 1 else parts[0]


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
        combined = combined[~combined.index.duplicated(keep="first")].sort_index()
        return combined

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self._target_cols = list(y.columns)
        self._X_train = X.copy()
        full_X = self._combine_X(None)
        idx, asset_cache, pair_cache, common_tail = _prep_caches(full_X, self._pairs_df)
        # Slice to y.index
        pos = idx.get_indexer(y.index)
        asset_cache = {k: v[pos] for k, v in asset_cache.items()}
        pair_cache = {k: v[pos] for k, v in pair_cache.items()}
        common_tail = common_tail[pos]

        params = dict(
            device="cuda",
            tree_method="hist",
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            verbosity=0,
            n_jobs=1,
        )
        jobs = []
        for tgt in self._target_cols:
            pair = self._pair_of[tgt]
            X_j = _feats_for_target(pair, asset_cache, pair_cache, common_tail)
            jobs.append((X_j, y[tgt].values, params))
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
            self._models = list(ex.map(_fit_one, jobs))

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self._target_cols is not None
        full_X = self._combine_X(X)
        idx, asset_cache, pair_cache, common_tail = _prep_caches(full_X, self._pairs_df)
        pos = idx.get_indexer(X.index)
        asset_cache = {k: v[pos] for k, v in asset_cache.items()}
        pair_cache = {k: v[pos] for k, v in pair_cache.items()}
        common_tail = common_tail[pos]

        jobs = []
        for j, tgt in enumerate(self._target_cols):
            pair = self._pair_of[tgt]
            X_j = _feats_for_target(pair, asset_cache, pair_cache, common_tail)
            jobs.append((X_j, self._models[j]))
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS_FIT) as ex:
            results = list(ex.map(_predict_one, jobs))
        out = np.zeros((len(X), len(self._target_cols)), dtype=np.float32)
        for j, p in enumerate(results):
            if p is not None:
                out[:, j] = p
        return pd.DataFrame(out, index=X.index, columns=self._target_cols)
