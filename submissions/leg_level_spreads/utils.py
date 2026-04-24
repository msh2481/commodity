"""Shared helpers for the leg-level-spreads thread.

Central convention (verified empirically against train_labels.csv):

    label[target, t, lag] = log(price[leg_a, t+1+lag] / price[leg_a, t+1])
                          - log(price[leg_b, t+1+lag] / price[leg_b, t+1])

So "leg return" for an asset at date t and horizon L is the forward L-day
log-return observed starting at t+1:

    leg_return(asset, t, L) = log(price[asset, t+1+L] / price[asset, t+1])

The path uses the next observation (t+1) as the entry price — so a
feature computed from information up to and including t is strictly
non-leaky with respect to this label.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
LAGS: tuple[int, ...] = (1, 2, 3, 4)


def load_target_pairs() -> pd.DataFrame:
    tp = pd.read_csv(DATA_DIR / "target_pairs.csv")
    legs_a = []
    legs_b = []
    for pair in tp["pair"]:
        parts = [p.strip() for p in pair.split(" - ")]
        legs_a.append(parts[0])
        legs_b.append(parts[1] if len(parts) == 2 else None)
    tp = tp.copy()
    tp["leg_a"] = legs_a
    tp["leg_b"] = legs_b
    return tp


def unique_legs(target_pairs: pd.DataFrame) -> list[str]:
    seen: set[str] = set()
    for col in ("leg_a", "leg_b"):
        for v in target_pairs[col].dropna():
            seen.add(v)
    return sorted(seen)


def leg_returns(prices: pd.DataFrame, legs: list[str], lags=LAGS) -> pd.DataFrame:
    """Build (date_id × (asset, lag)) forward log-returns.

    Returns a DataFrame with MultiIndex columns (asset, lag).
    """
    p = prices[legs]
    out = {}
    # entry price is price at t+1
    p_entry = p.shift(-1)
    for L in lags:
        p_exit = p.shift(-(1 + L))
        r = np.log(p_exit / p_entry)
        for asset in legs:
            out[(asset, L)] = r[asset]
    df = pd.DataFrame(out)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["asset", "lag"])
    return df


def compose_targets(
    leg_preds: pd.DataFrame,
    target_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Compose predicted target columns from predicted leg returns.

    leg_preds: DataFrame indexed by date_id, columns MultiIndex (asset, lag).
    target_pairs: output of load_target_pairs().

    Returns: DataFrame indexed by date_id with columns target_0..target_423.
    """
    target_cols = list(target_pairs["target"])
    out = pd.DataFrame(0.0, index=leg_preds.index, columns=target_cols)
    for _, row in target_pairs.iterrows():
        lag = int(row["lag"])
        a = row["leg_a"]
        tgt = row["target"]
        val = leg_preds[(a, lag)]
        b = row["leg_b"]
        if b is not None and isinstance(b, str):
            val = val - leg_preds[(b, lag)]
        out[tgt] = val
    return out
