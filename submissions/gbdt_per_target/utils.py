"""Shared feature-engineering helpers for gbdt_per_target."""
from __future__ import annotations

import numpy as np
import pandas as pd


def price_columns(X: pd.DataFrame) -> list[str]:
    cols = list(X.columns)
    lme = [c for c in cols if c.startswith("LME_") and c.endswith("_Close")]
    jpx_close = [c for c in cols if c.startswith("JPX_") and c.endswith("_Close")]
    us_close = [c for c in cols if c.startswith("US_Stock_") and c.endswith("_adj_close")]
    fx = [c for c in cols if c.startswith("FX_")]
    return lme + jpx_close + us_close + fx


def build_feature_matrix(
    X: pd.DataFrame,
    ret_lags: tuple[int, ...] = (1, 5, 20),
    vol_windows: tuple[int, ...] = (20,),
    mean_windows: tuple[int, ...] = (20,),
    add_xs_rank_1d: bool = True,
    add_calendar: bool = True,
) -> pd.DataFrame:
    """Compute features from raw prices. Index preserved (sorted).

    Features per asset:
    - log_return at each lag in ret_lags
    - rolling std(log_return_1) at each window in vol_windows
    - rolling mean(log_return_1) at each window in mean_windows
    - cross-sectional rank across assets of 1-day log-return (if add_xs_rank_1d)

    Plus optional calendar features (dow, month-of-quarter) derived from date_id.
    NaNs are preserved — XGBoost handles them natively.
    """
    Xs = X.sort_index()
    price_cols = price_columns(Xs)
    prices = Xs[price_cols].astype(np.float64)
    # ffill to smooth over missing exchange days
    prices = prices.ffill()
    logp = np.log(prices.clip(lower=1e-12))
    ret1 = logp.diff()

    parts = []
    for lag in ret_lags:
        r = (logp - logp.shift(lag)).add_suffix(f"_logret{lag}")
        parts.append(r)

    for w in vol_windows:
        v = ret1.rolling(window=w, min_periods=max(5, w // 2)).std().add_suffix(f"_vol{w}")
        parts.append(v)

    for w in mean_windows:
        m = ret1.rolling(window=w, min_periods=max(5, w // 2)).mean().add_suffix(f"_mean{w}")
        parts.append(m)

    if add_xs_rank_1d:
        # Cross-sectional percentile rank of 1-day return across assets
        xs = ret1.rank(axis=1, pct=True).add_suffix("_xsrank1")
        parts.append(xs)

    if add_calendar:
        date_id = Xs.index.to_numpy()
        cal = pd.DataFrame(
            {
                "cal_dow": (date_id % 5).astype(np.float32),
                "cal_wom": (date_id % 22).astype(np.float32),
                "cal_trend": (date_id / 2000.0).astype(np.float32),
            },
            index=Xs.index,
        )
        parts.append(cal)

    feats = pd.concat(parts, axis=1)

    return feats.astype(np.float32)
