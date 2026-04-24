"""Feature builders for the leg-level-spreads thread.

All features are computed over a price panel indexed by date_id. A
feature value at row t uses only information with index <= t (strict
non-leakage w.r.t. labels, which use prices at t+1 and beyond).

Missing prices are forward-filled (carry-forward) before feature
derivation; returns on gap days become zero. Residual NaN feature cells
are replaced with 0.0 by the caller.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ffill_log(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices.ffill())


def minimal_features(prices: pd.DataFrame, legs: list[str]) -> pd.DataFrame:
    """Lagged log-returns at 1, 5, 20 days for each leg asset.

    Columns look like "ret1_<asset>", "ret5_<asset>", "ret20_<asset>".
    Returned DataFrame is indexed by date_id with the same index as
    `prices`. NaN cells are left for the caller to impute.
    """
    logp = _ffill_log(prices[legs])
    cols = {}
    for k in (1, 5, 20):
        r = logp - logp.shift(k)
        for a in legs:
            cols[f"ret{k}_{a}"] = r[a]
    return pd.DataFrame(cols, index=prices.index)


def price_like_columns(all_cols: list[str]) -> list[str]:
    """Return names of columns that look like price/FX level series (not
    volume or open-interest). We take log-returns of these."""
    out = []
    for c in all_cols:
        cl = c.lower()
        if "volume" in cl or "open_interest" in cl:
            continue
        out.append(c)
    return out


def wide_plus_features(prices: pd.DataFrame, legs: list[str]) -> pd.DataFrame:
    """Wide features + more lag horizons + more cross-sectional ranks +
    rolling-Sharpe-like features. Intended as a richer feature matrix
    for ensemble members that can absorb more dimensions.
    """
    all_cols = list(prices.columns)
    price_cols = price_like_columns(all_cols)

    logp = _ffill_log(prices[price_cols])
    frames = []

    # extended horizons for all price cols: 1, 2, 3, 5, 10, 20, 60
    for k in (1, 2, 3, 5, 10, 20, 60):
        r = logp - logp.shift(k)
        r.columns = [f"ret{k}_{c}" for c in price_cols]
        frames.append(r)

    # leg-specific moments
    leg_logp = logp[legs]
    leg_r1 = leg_logp - leg_logp.shift(1)

    for w in (5, 20, 60):
        v = leg_r1.rolling(w, min_periods=max(3, w // 3)).std()
        v.columns = [f"vol{w}_{a}" for a in legs]
        frames.append(v)

    for w in (5, 20, 60):
        m = leg_r1.rolling(w, min_periods=max(3, w // 3)).mean()
        m.columns = [f"mom{w}_{a}" for a in legs]
        frames.append(m)

    # realized-Sharpe-like: mean / std
    for w in (20, 60):
        m = leg_r1.rolling(w, min_periods=max(3, w // 3)).mean()
        s = leg_r1.rolling(w, min_periods=max(3, w // 3)).std()
        sr = m.div(s.replace(0, np.nan))
        sr.columns = [f"sr{w}_{a}" for a in legs]
        frames.append(sr)

    # cross-sectional ranks per date at multiple horizons
    for k in (1, 5, 20, 60):
        r = leg_logp - leg_logp.shift(k)
        xr = r.rank(axis=1, pct=True) * 2 - 1
        xr.columns = [f"xrank{k}_{a}" for a in legs]
        frames.append(xr)

    # market proxy factor + beta
    mkt = leg_r1.mean(axis=1)
    for w in (20, 60):
        cov = leg_r1.rolling(w, min_periods=max(3, w // 3)).cov(mkt)
        var = mkt.rolling(w, min_periods=max(3, w // 3)).var()
        beta = cov.div(var, axis=0)
        beta.columns = [f"beta{w}_{a}" for a in legs]
        frames.append(beta)

    # bucket-mean factor returns
    def bucket(c: str) -> str:
        if c.startswith("LME_"):
            return "LME"
        if c.startswith("JPX_"):
            return "JPX"
        if c.startswith("US_Stock_"):
            return "US"
        if c.startswith("FX_"):
            return "FX"
        return "OTHER"

    buckets = {b: [c for c in price_cols if bucket(c) == b] for b in ("LME", "JPX", "US", "FX")}
    factor_frames = {}
    for b, cols in buckets.items():
        if not cols:
            continue
        for k in (1, 5, 20):
            r = (logp[cols] - logp[cols].shift(k)).mean(axis=1)
            factor_frames[f"factor_ret{k}_{b}"] = r
    frames.append(pd.DataFrame(factor_frames, index=prices.index))

    dow = pd.get_dummies(prices.index.to_series() % 5, prefix="dow").astype(float)
    dow.index = prices.index
    frames.append(dow)

    return pd.concat(frames, axis=1)


def pair_rank_features(
    prices: pd.DataFrame, target_pairs: pd.DataFrame
) -> pd.DataFrame:
    """Cross-sectional rank of pair log-spread z-scores per date. Gives
    a per-pair "how extreme is this spread today compared to all others"
    signal.
    """
    pair_set = set()
    for _, row in target_pairs.iterrows():
        b = row["leg_b"]
        if isinstance(b, str) and b:
            pair_set.add((row["leg_a"], b))
    pairs = sorted(pair_set)
    if not pairs:
        return pd.DataFrame(index=prices.index)
    logp = _ffill_log(prices)
    a_names = [f"{a}__{b}" for a, b in pairs]
    logA = pd.DataFrame({n: logp[a] for n, (a, b) in zip(a_names, pairs)}, index=prices.index)
    logB = pd.DataFrame({n: logp[b] for n, (a, b) in zip(a_names, pairs)}, index=prices.index)
    spread = logA - logB
    m60 = spread.rolling(60, min_periods=20).mean()
    s60 = spread.rolling(60, min_periods=20).std()
    z = (spread - m60).div(s60.replace(0, np.nan))
    # cross-sectional rank per date, scaled to [-1, 1]
    xrank = z.rank(axis=1, pct=True) * 2 - 1
    xrank.columns = [f"pair_xrank60_{n}" for n in a_names]
    return xrank


def volume_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Volume-like features: log(volume+1) z-scored, and 5-day vol-change."""
    cols = [c for c in prices.columns if "volume" in c.lower() or "open_interest" in c.lower()]
    if not cols:
        return pd.DataFrame(index=prices.index)
    v = prices[cols].ffill().fillna(0)
    logv = np.log1p(v)
    # rolling 20-day z-score
    m = logv.rolling(20, min_periods=5).mean()
    s = logv.rolling(20, min_periods=5).std()
    z = (logv - m).div(s.replace(0, np.nan))
    z.columns = [f"vol_z20_{c}" for c in cols]
    # 5-day log-change of volume
    dv = logv - logv.shift(5)
    dv.columns = [f"vol_d5_{c}" for c in cols]
    return pd.concat([z, dv], axis=1)


def pair_spread_features(
    prices: pd.DataFrame, target_pairs: pd.DataFrame
) -> pd.DataFrame:
    """Features built from the specific leg pairs that appear in the
    competition targets. For each unique (A, B) pair:

      - log_spread = log(price_A) - log(price_B) at various rolling
        windows expressed as z-scores (mean-reversion signal)
      - spread_return at 1, 5, 20 days (pair momentum)

    Uses only pairs that show up in target_pairs.csv (de-duplicated,
    ignoring lag), so the feature matrix stays bounded (~85 unique
    pairs × 6 features ≈ 500 cols).
    """
    pair_set = set()
    for _, row in target_pairs.iterrows():
        b = row["leg_b"]
        if isinstance(b, str) and b:
            pair_set.add((row["leg_a"], b))
    pairs = sorted(pair_set)

    logp = _ffill_log(prices)
    frames = []

    a_names = [f"{a}__{b}" for a, b in pairs]
    logA = pd.DataFrame({f"{a}__{b}": logp[a] for a, b in pairs}, index=prices.index)
    logB = pd.DataFrame({f"{a}__{b}": logp[b] for a, b in pairs}, index=prices.index)

    spread = logA - logB
    # pair log-spread z-score over rolling windows (mean-reversion)
    for w in (20, 60):
        m = spread.rolling(w, min_periods=max(3, w // 3)).mean()
        s = spread.rolling(w, min_periods=max(3, w // 3)).std()
        z = (spread - m).div(s.replace(0, np.nan))
        z.columns = [f"pair_sprz{w}_{c}" for c in a_names]
        frames.append(z)

    # pair return differences at multiple horizons (covers all 4 label lags)
    for k in (1, 2, 3, 5, 20):
        sr = spread - spread.shift(k)
        sr.columns = [f"pair_dret{k}_{c}" for c in a_names]
        frames.append(sr)

    # realized spread vol (20-day)
    sr1 = spread - spread.shift(1)
    sv = sr1.rolling(20, min_periods=7).std()
    sv.columns = [f"pair_vol20_{c}" for c in a_names]
    frames.append(sv)

    return pd.concat(frames, axis=1)


def wide_features(prices: pd.DataFrame, legs: list[str]) -> pd.DataFrame:
    """Much broader feature set using ALL price-like columns in
    train.csv (not just legs). For each price-like column include ret1,
    ret5, ret20. For legs, additionally include ret60, vol20, mom20,
    xrank5, beta20. Plus calendar dummies and bucket-mean factor returns.
    """
    all_cols = list(prices.columns)
    price_cols = price_like_columns(all_cols)

    logp = _ffill_log(prices[price_cols])
    frames = []

    # universal: 1/5/20-day returns for every price-like column
    for k in (1, 5, 20):
        r = logp - logp.shift(k)
        r.columns = [f"ret{k}_{c}" for c in price_cols]
        frames.append(r)

    # leg-only extras
    leg_logp = logp[legs]
    leg_r1 = leg_logp - leg_logp.shift(1)

    r60 = leg_logp - leg_logp.shift(60)
    r60.columns = [f"ret60_{a}" for a in legs]
    frames.append(r60)

    vol20 = leg_r1.rolling(20, min_periods=7).std()
    vol20.columns = [f"vol20_{a}" for a in legs]
    frames.append(vol20)

    mom20 = leg_r1.rolling(20, min_periods=7).mean()
    mom20.columns = [f"mom20_{a}" for a in legs]
    frames.append(mom20)

    r5_leg = leg_logp - leg_logp.shift(5)
    xrank5 = r5_leg.rank(axis=1, pct=True) * 2 - 1
    xrank5.columns = [f"xrank5_{a}" for a in legs]
    frames.append(xrank5)

    # market proxy (equal-weight mean of leg 1-day returns)
    mkt = leg_r1.mean(axis=1)
    cov20 = leg_r1.rolling(20, min_periods=7).cov(mkt)
    var20 = mkt.rolling(20, min_periods=7).var()
    beta20 = cov20.div(var20, axis=0)
    beta20.columns = [f"beta20_{a}" for a in legs]
    frames.append(beta20)

    # bucket-mean factor returns
    def bucket(c: str) -> str:
        if c.startswith("LME_"):
            return "LME"
        if c.startswith("JPX_"):
            return "JPX"
        if c.startswith("US_Stock_"):
            return "US"
        if c.startswith("FX_"):
            return "FX"
        return "OTHER"

    buckets = {b: [c for c in price_cols if bucket(c) == b] for b in ("LME", "JPX", "US", "FX")}
    logp_b = logp
    factor_frames = {}
    for b, cols in buckets.items():
        if not cols:
            continue
        r1 = (logp_b[cols] - logp_b[cols].shift(1)).mean(axis=1)
        r5 = (logp_b[cols] - logp_b[cols].shift(5)).mean(axis=1)
        factor_frames[f"factor_ret1_{b}"] = r1
        factor_frames[f"factor_ret5_{b}"] = r5
    frames.append(pd.DataFrame(factor_frames, index=prices.index))

    # calendar dummies
    dow = pd.get_dummies(prices.index.to_series() % 5, prefix="dow").astype(float)
    dow.index = prices.index
    frames.append(dow)

    return pd.concat(frames, axis=1)


def extended_features(prices: pd.DataFrame, legs: list[str]) -> pd.DataFrame:
    """Richer feature set: returns at multiple horizons, rolling vol,
    rolling mean, cross-sectional rank (short-horizon return), market
    beta, calendar indicators.
    """
    logp = _ffill_log(prices[legs])
    frames = []

    # returns at 1, 5, 20, 60
    for k in (1, 5, 20, 60):
        r = logp - logp.shift(k)
        r.columns = [f"ret{k}_{a}" for a in legs]
        frames.append(r)

    # rolling vol of 1-day returns at 5, 20, 60
    r1 = logp - logp.shift(1)
    for w in (5, 20, 60):
        v = r1.rolling(w, min_periods=max(3, w // 3)).std()
        v.columns = [f"vol{w}_{a}" for a in legs]
        frames.append(v)

    # rolling mean of 1-day returns at 5, 20, 60 (momentum)
    for w in (5, 20, 60):
        m = r1.rolling(w, min_periods=max(3, w // 3)).mean()
        m.columns = [f"mom{w}_{a}" for a in legs]
        frames.append(m)

    # cross-sectional rank of 5-day return per date, scaled to [-1, 1]
    r5 = logp - logp.shift(5)
    r5_rank = r5.rank(axis=1, pct=True) * 2 - 1
    r5_rank.columns = [f"xrank5_{a}" for a in legs]
    frames.append(r5_rank)

    # market proxy = equal-weighted mean of all leg 1-day returns
    mkt = r1.mean(axis=1)
    for w in (20, 60):
        cov = r1.rolling(w, min_periods=max(3, w // 3)).cov(mkt)
        var = mkt.rolling(w, min_periods=max(3, w // 3)).var()
        beta = cov.div(var, axis=0)
        beta.columns = [f"beta{w}_{a}" for a in legs]
        frames.append(beta)

    # calendar: approximate day-of-week using date_id mod 5 (no calendar given)
    dow = pd.get_dummies(prices.index.to_series() % 5, prefix="dow").astype(float)
    dow.index = prices.index
    frames.append(dow)

    return pd.concat(frames, axis=1)
