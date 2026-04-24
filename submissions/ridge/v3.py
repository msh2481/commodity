"""v3: per-target ridge with spread-specific features.

For each of 424 targets, fit an independent ridge on:
  - leg A block: log-returns {1,5,20,60}, vol {5,20,60}, mret {5,20,60}
  - leg B block (if spread): same
  - spread block (if spread): spread-level z-score {20,60}, spread return {1,5,20}, spread vol {20,60}
  - market context: median |1d return| at {5,20}, cross-sectional std of returns at {5,20}

NaN handling: drop rows where target y is NaN from training; zero-impute feature NaN.
All features computed from full X panel (causal).
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


ALPHA = 10.0
RET_LAGS = [1, 5, 20, 60]
VOL_WINDOWS = [5, 20, 60]
MRET_WINDOWS = [5, 20, 60]
SPREAD_Z_WINDOWS = [20, 60]
SPREAD_RET_WINDOWS = [1, 5, 20]
SPREAD_VOL_WINDOWS = [20, 60]
MARKET_WINDOWS = [5, 20]


_CACHE: dict[str, object] = {}


def _asset_feats(log_p: pd.Series) -> pd.DataFrame:
    r1 = log_p.diff(1)
    parts = {}
    for lag in RET_LAGS:
        parts[f"lret{lag}"] = log_p.diff(lag)
    for w in VOL_WINDOWS:
        parts[f"vol{w}"] = r1.rolling(w, min_periods=max(2, w // 2)).std()
    for w in MRET_WINDOWS:
        parts[f"mret{w}"] = r1.rolling(w, min_periods=max(2, w // 2)).mean()
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
    return pd.DataFrame(parts)


def _market_feats(log_p: pd.DataFrame) -> pd.DataFrame:
    r1 = log_p.diff(1)
    parts = {}
    for w in MARKET_WINDOWS:
        parts[f"mkt_abs_ret{w}"] = r1.abs().mean(axis=1).rolling(w, min_periods=max(2, w // 2)).mean()
        parts[f"mkt_xs_std{w}"] = r1.std(axis=1).rolling(w, min_periods=max(2, w // 2)).mean()
    return pd.DataFrame(parts)


def _prepare() -> None:
    """Build and cache all feature blocks keyed by asset and pair."""
    if "ready" in _CACHE:
        return
    X = load_raw_X()
    pairs = pd.read_csv(DATA_DIR / "target_pairs.csv")

    legs = set()
    for p in pairs.pair:
        for a in p.split(" - "):
            legs.add(a.strip())
    legs = sorted(legs)

    # per-leg log-prices
    log_p_by_leg: dict[str, pd.Series] = {}
    for leg in legs:
        s = X[leg].astype(np.float64).ffill()
        log_p_by_leg[leg] = np.log(s.where(s > 0))

    # per-leg features
    leg_feats: dict[str, pd.DataFrame] = {}
    for leg, lp in log_p_by_leg.items():
        leg_feats[leg] = _asset_feats(lp)

    # per-pair spread features (pairs that are spreads)
    pair_feats: dict[str, pd.DataFrame] = {}
    for p in pairs.pair.unique():
        if " - " in p:
            a, b = [x.strip() for x in p.split(" - ")]
            pair_feats[p] = _spread_feats(log_p_by_leg[a], log_p_by_leg[b])

    # market features built on all leg prices
    log_p_all = pd.DataFrame({k: v for k, v in log_p_by_leg.items()})
    mkt = _market_feats(log_p_all)

    _CACHE["leg_feats"] = leg_feats
    _CACHE["pair_feats"] = pair_feats
    _CACHE["mkt"] = mkt
    _CACHE["pairs"] = pairs.set_index("target")
    _CACHE["date_index"] = X.index
    _CACHE["ready"] = True


def _build_target_features(target: str) -> pd.DataFrame:
    _prepare()
    pairs = _CACHE["pairs"]
    leg_feats: dict = _CACHE["leg_feats"]
    pair_feats: dict = _CACHE["pair_feats"]
    mkt: pd.DataFrame = _CACHE["mkt"]

    row = pairs.loc[target]
    pair = row["pair"]
    parts = []
    if " - " in pair:
        a, b = [x.strip() for x in pair.split(" - ")]
        fa = leg_feats[a].add_prefix("A_")
        fb = leg_feats[b].add_prefix("B_")
        fs = pair_feats[pair]
        parts.extend([fa, fb, fs])
    else:
        fa = leg_feats[pair].add_prefix("A_")
        parts.append(fa)
    parts.append(mkt)
    feat = pd.concat(parts, axis=1)
    return feat.replace([np.inf, -np.inf], np.nan)


class Model:
    device = "cpu"

    def __init__(self) -> None:
        self.alpha = ALPHA
        self._target_cols: list[str] | None = None
        self._per_target: dict[str, dict] = {}

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        _prepare()
        self._target_cols = list(y.columns)
        train_index = X.index
        for t in self._target_cols:
            feats_full = _build_target_features(t)
            feat_train = feats_full.loc[train_index]
            y_t = y[t].loc[train_index]
            mask = y_t.notna() & feat_train.notna().all(axis=1)
            if mask.sum() < 50:
                mask = y_t.notna()
                Xf = feat_train.loc[mask].fillna(0.0).values
            else:
                Xf = feat_train.loc[mask].values
            yf = y_t.loc[mask].values.astype(np.float64)
            mu = Xf.mean(axis=0)
            sd = Xf.std(axis=0)
            sd = np.where(sd < 1e-12, 1.0, sd)
            Xs = (Xf - mu) / sd
            reg = Ridge(alpha=self.alpha, fit_intercept=True)
            reg.fit(Xs, yf)
            self._per_target[t] = {
                "reg": reg,
                "mu": mu,
                "sd": sd,
                "cols": list(feat_train.columns),
            }

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self._target_cols is not None
        test_index = X.index
        out = np.zeros((len(test_index), len(self._target_cols)), dtype=np.float64)
        for j, t in enumerate(self._target_cols):
            spec = self._per_target[t]
            feats = _build_target_features(t).loc[test_index].reindex(columns=spec["cols"])
            Xf = feats.fillna(0.0).values
            Xs = (Xf - spec["mu"]) / spec["sd"]
            Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
            out[:, j] = spec["reg"].predict(Xs)
        return pd.DataFrame(out, index=test_index, columns=self._target_cols)
