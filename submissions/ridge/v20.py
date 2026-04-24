"""v20: v14 (feature selection + rank-y) with expanded base feature set.

Adds to v4's feature set:
  - sret at {2, 3, 4, 10, 40} to match target lags better
  - szs at window 120 (longer-horizon mean reversion)
  - xsr at {2, 3, 4, 10, 60} for more cross-sectional horizons

Feature selection (top-K by |corr| with y_rank) filters the bigger pool.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from utils import DATA_DIR, load_raw_X


ALPHA = 0.3
TOP_K = 35
RET_LAGS = [1, 2, 3, 4, 5, 10, 20, 40, 60]
VOL_WINDOWS = [5, 20, 60]
MRET_WINDOWS = [5, 20, 60]
XS_RANK_RET_LAGS = [1, 2, 3, 4, 5, 10, 20, 60]
XS_RANK_VOL_WINDOWS = [20, 60]
SPREAD_Z_WINDOWS = [10, 20, 60, 120]
SPREAD_RET_WINDOWS = [1, 2, 3, 4, 5, 10, 20, 40]
SPREAD_VOL_WINDOWS = [20, 60]
MARKET_WINDOWS = [5, 20]


_CACHE: dict[str, object] = {}


def _asset_feats(log_p: pd.Series, xs_blocks: dict[str, pd.Series]) -> pd.DataFrame:
    r1 = log_p.diff(1)
    parts = {}
    for lag in RET_LAGS:
        parts[f"lret{lag}"] = log_p.diff(lag)
    for w in VOL_WINDOWS:
        parts[f"vol{w}"] = r1.rolling(w, min_periods=max(2, w // 2)).std()
    for w in MRET_WINDOWS:
        parts[f"mret{w}"] = r1.rolling(w, min_periods=max(2, w // 2)).mean()
    parts["abs_r1"] = r1.abs()
    for k, s in xs_blocks.items():
        parts[k] = s
    return pd.DataFrame(parts)


def _spread_feats(log_a: pd.Series, log_b: pd.Series) -> pd.DataFrame:
    spread = log_a - log_b
    r1 = spread.diff(1)
    parts = {}
    for w in SPREAD_Z_WINDOWS:
        m = spread.rolling(w, min_periods=max(2, w // 2)).mean()
        s = spread.rolling(w, min_periods=max(2, w // 2)).std()
        parts[f"szs{w}"] = (spread - m) / s.replace(0, np.nan)
    for w in SPREAD_RET_WINDOWS:
        parts[f"sret{w}"] = spread.diff(w)
    for w in SPREAD_VOL_WINDOWS:
        parts[f"svol{w}"] = r1.rolling(w, min_periods=max(2, w // 2)).std()
    parts["sacc"] = spread.diff(1) - spread.diff(5) / 5
    return pd.DataFrame(parts)


def _market_feats(log_p: pd.DataFrame) -> pd.DataFrame:
    r1 = log_p.diff(1)
    parts = {}
    for w in MARKET_WINDOWS:
        parts[f"mkt_abs_ret{w}"] = r1.abs().mean(axis=1).rolling(w, min_periods=max(2, w // 2)).mean()
        parts[f"mkt_xs_std{w}"] = r1.std(axis=1).rolling(w, min_periods=max(2, w // 2)).mean()
    return pd.DataFrame(parts)


def _prepare() -> None:
    if "ready" in _CACHE:
        return
    X = load_raw_X()
    pairs = pd.read_csv(DATA_DIR / "target_pairs.csv")

    legs = set()
    for p in pairs.pair:
        for a in p.split(" - "):
            legs.add(a.strip())
    legs = sorted(legs)

    log_p_by_leg: dict[str, pd.Series] = {}
    for leg in legs:
        s = X[leg].astype(np.float64).ffill()
        log_p_by_leg[leg] = np.log(s.where(s > 0))
    log_p_all = pd.DataFrame(log_p_by_leg)

    xs_rank_ret: dict[int, pd.DataFrame] = {}
    for lag in XS_RANK_RET_LAGS:
        rL = log_p_all.diff(lag)
        xs_rank_ret[lag] = rL.rank(axis=1, pct=True) - 0.5
    r1_all = log_p_all.diff(1)
    xs_rank_vol: dict[int, pd.DataFrame] = {}
    for w in XS_RANK_VOL_WINDOWS:
        v = r1_all.rolling(w, min_periods=max(2, w // 2)).std()
        xs_rank_vol[w] = v.rank(axis=1, pct=True) - 0.5

    leg_feats: dict[str, pd.DataFrame] = {}
    for leg in legs:
        xs_blocks: dict[str, pd.Series] = {}
        for lag in XS_RANK_RET_LAGS:
            xs_blocks[f"xsr{lag}"] = xs_rank_ret[lag][leg]
        for w in XS_RANK_VOL_WINDOWS:
            xs_blocks[f"xsvol{w}"] = xs_rank_vol[w][leg]
        leg_feats[leg] = _asset_feats(log_p_by_leg[leg], xs_blocks)

    pair_feats: dict[str, pd.DataFrame] = {}
    for p in pairs.pair.unique():
        if " - " in p:
            a, b = [x.strip() for x in p.split(" - ")]
            pair_feats[p] = _spread_feats(log_p_by_leg[a], log_p_by_leg[b])

    mkt = _market_feats(log_p_all)

    _CACHE["leg_feats"] = leg_feats
    _CACHE["pair_feats"] = pair_feats
    _CACHE["mkt"] = mkt
    _CACHE["pairs"] = pairs.set_index("target")
    _CACHE["ready"] = True


def _build_target_features(target: str) -> pd.DataFrame:
    _prepare()
    row = _CACHE["pairs"].loc[target]
    pair = row["pair"]
    leg_feats = _CACHE["leg_feats"]
    pair_feats = _CACHE["pair_feats"]
    mkt = _CACHE["mkt"]

    parts = []
    if " - " in pair:
        a, b = [x.strip() for x in pair.split(" - ")]
        parts.append(leg_feats[a].add_prefix("A_"))
        parts.append(leg_feats[b].add_prefix("B_"))
        parts.append(pair_feats[pair])
    else:
        parts.append(leg_feats[pair].add_prefix("A_"))
    parts.append(mkt)
    feat = pd.concat(parts, axis=1)
    return feat.replace([np.inf, -np.inf], np.nan)


class Model:
    device = "cpu"

    def __init__(self) -> None:
        self.alpha = ALPHA
        self._target_cols: list[str] | None = None
        self._per_target: dict[str, dict] = {}

    @staticmethod
    def _xs_rank_targets(y: pd.DataFrame) -> pd.DataFrame:
        return y.rank(axis=1, pct=True) - 0.5

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        _prepare()
        self._target_cols = list(y.columns)
        y_rank = self._xs_rank_targets(y)
        train_index = X.index
        for t in self._target_cols:
            feats_full = _build_target_features(t)
            feat_train = feats_full.loc[train_index]
            y_t = y_rank[t].loc[train_index]
            mask = y_t.notna()
            Xf = feat_train.loc[mask].fillna(0.0).values.astype(np.float64)
            yf = y_t.loc[mask].values.astype(np.float64)
            cols = list(feat_train.columns)
            n_feats = Xf.shape[1]
            k = min(TOP_K, n_feats)
            yf_c = yf - yf.mean()
            y_std = yf_c.std()
            if y_std < 1e-12:
                sel = np.arange(n_feats)
            else:
                X_c = Xf - Xf.mean(axis=0, keepdims=True)
                x_std = X_c.std(axis=0)
                x_std = np.where(x_std < 1e-12, 1.0, x_std)
                corr = (X_c * yf_c[:, None]).mean(axis=0) / (x_std * y_std)
                sel = np.argsort(-np.abs(corr))[:k]
            sel = np.sort(sel)
            Xf_sel = Xf[:, sel]
            mu = Xf_sel.mean(axis=0)
            sd = Xf_sel.std(axis=0)
            sd = np.where(sd < 1e-12, 1.0, sd)
            Xs = (Xf_sel - mu) / sd
            reg = Ridge(alpha=self.alpha, fit_intercept=True)
            reg.fit(Xs, yf)
            self._per_target[t] = {
                "reg": reg, "mu": mu, "sd": sd, "cols": cols, "sel": sel.astype(np.int32),
            }

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self._target_cols is not None
        test_index = X.index
        out = np.zeros((len(test_index), len(self._target_cols)), dtype=np.float64)
        for j, t in enumerate(self._target_cols):
            spec = self._per_target[t]
            feats = _build_target_features(t).loc[test_index].reindex(columns=spec["cols"])
            Xf = feats.fillna(0.0).values.astype(np.float64)
            Xf = Xf[:, spec["sel"]]
            Xs = (Xf - spec["mu"]) / spec["sd"]
            Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
            out[:, j] = spec["reg"].predict(Xs)
        return pd.DataFrame(out, index=test_index, columns=self._target_cols)
