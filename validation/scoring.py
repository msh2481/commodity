"""Shared scoring helpers for the Mitsui validator."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def per_date_spearman(preds: pd.DataFrame, truth: pd.DataFrame) -> pd.Series:
    """Cross-sectional Spearman correlation per date_id.

    Both inputs must be indexed by date_id with identical target columns.
    NaN targets (or NaN predictions) on a given date are dropped before the
    correlation is computed. Dates with fewer than 2 usable targets return NaN.
    """
    assert preds.index.name == "date_id" and truth.index.name == "date_id"
    assert list(preds.columns) == list(truth.columns)
    common = preds.index.intersection(truth.index)
    preds = preds.loc[common]
    truth = truth.loc[common]

    out = {}
    for date_id in common:
        p = preds.loc[date_id].values
        t = truth.loc[date_id].values
        mask = ~(np.isnan(p) | np.isnan(t))
        if mask.sum() < 2:
            out[date_id] = np.nan
            continue
        rho, _ = spearmanr(p[mask], t[mask])
        out[date_id] = rho
    return pd.Series(out, name="rho").rename_axis("date_id")


def sharpe_from_rhos(rhos: pd.Series) -> float:
    """Sharpe-like metric: mean/std of per-date Spearman correlations.

    NaN rhos are dropped before aggregation.
    """
    rhos = rhos.dropna()
    if len(rhos) < 2:
        return float("nan")
    std = rhos.std(ddof=1)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(rhos.mean() / std)
