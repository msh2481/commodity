"""Shared feature-engineering helpers for the ridge thread.

All functions are strictly causal: feature(d) uses only X at dates <= d.
Features are computed once on the full X panel; fit/predict just look up rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
FX_PREFIX = "FX_"
LME_SUFFIX = "_Close"
US_CLOSE_SUFFIX = "adj_close"
JPX_CLOSE_SUFFIX = "_Close"
JPX_VOL_SUFFIX = "_Volume"
US_VOL_SUFFIX = "adj_volume"


def load_raw_X() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "train.csv").set_index("date_id").sort_index()


def price_columns(df: pd.DataFrame) -> list[str]:
    """All columns that look like a price series we can log-return."""
    cols = []
    for c in df.columns:
        if c.startswith(FX_PREFIX):
            cols.append(c)
        elif c.startswith("LME_") and c.endswith(LME_SUFFIX):
            cols.append(c)
        elif c.startswith("JPX_") and c.endswith(JPX_CLOSE_SUFFIX):
            cols.append(c)
        elif c.startswith("US_") and c.endswith(US_CLOSE_SUFFIX):
            cols.append(c)
    return cols


def volume_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c.startswith("JPX_") and c.endswith(JPX_VOL_SUFFIX):
            cols.append(c)
        elif c.startswith("US_") and c.endswith(US_VOL_SUFFIX):
            cols.append(c)
    return cols


def log_returns(prices: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    prices = prices.ffill()
    log_p = np.log(prices.where(prices > 0))
    parts = []
    for lag in lags:
        r = log_p.diff(lag)
        r.columns = [f"{c}__lret{lag}" for c in r.columns]
        parts.append(r)
    return pd.concat(parts, axis=1)


def rolling_vol(log_p: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Rolling std of 1-day log returns."""
    r1 = log_p.diff(1)
    parts = []
    for w in windows:
        s = r1.rolling(w, min_periods=max(2, w // 2)).std()
        s.columns = [f"{c}__vol{w}" for c in s.columns]
        parts.append(s)
    return pd.concat(parts, axis=1)


def rolling_mean_return(log_p: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Rolling mean of 1-day log returns."""
    r1 = log_p.diff(1)
    parts = []
    for w in windows:
        s = r1.rolling(w, min_periods=max(2, w // 2)).mean()
        s.columns = [f"{c}__mret{w}" for c in s.columns]
        parts.append(s)
    return pd.concat(parts, axis=1)


def cross_sectional_rank(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Per-row rank of each column, scaled to [-0.5, 0.5]."""
    ranks = df.rank(axis=1, pct=True) - 0.5
    ranks.columns = [f"{c}__{suffix}" for c in ranks.columns]
    return ranks


def clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Xs = (X - mu) / sd
    return Xs, mu, sd


def standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd
