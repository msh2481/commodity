from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def _iqm(values: np.ndarray) -> float:
    """Return the interquartile mean of a 1D finite array."""
    values = np.sort(np.asarray(values, dtype=float))
    trim = len(values) // 4
    if trim == 0:
        return float(values.mean())
    return float(values[trim:-trim].mean())


def _mean_abs_deviation_from_iqm(values: np.ndarray) -> float:
    center = _iqm(values)
    return float(np.mean(np.abs(np.asarray(values, dtype=float) - center)))


def pca_find(*arrays: np.ndarray | pd.Series) -> np.ndarray:
    """Return OLS coefs for predicting the last array from the preceding arrays."""
    assert len(arrays) >= 2
    x = np.column_stack([np.asarray(arr, dtype=float) for arr in arrays])
    assert x.ndim == 2
    assert np.isfinite(x).all()
    x -= x.mean(axis=0)
    std = x.std(axis=0)
    assert (std > 0).all()
    x /= std

    model = LinearRegression(fit_intercept=False)
    model.fit(x[:, :-1], x[:, -1])
    return model.coef_


def process_fx(df: pd.DataFrame, return_residuals: bool = False) -> pd.DataFrame:
    """Infer per-currency valuations from overdetermined FX pairs.

    For each column named like ``FX_EURUSD``, the model is:

        log(FX_EURUSD) = log(FX_EUR) - log(FX_USD) + residual

    The per-row least-squares solution is pinned by requiring the sum of
    currency log-valuations to be zero. Returned ``FX_<CCY>`` columns are
    valuations in level space, and ``FXR_<PAIR>`` columns are signed log errors:
    observed log pair minus predicted log pair.
    """
    fx_cols = [
        col
        for col in df.columns
        if isinstance(col, str) and col.startswith("FX_") and len(col) == 9
    ]

    pairs = [(col, col[3:6], col[6:9]) for col in fx_cols]
    currencies = sorted({ccy for _, base, quote in pairs for ccy in (base, quote)})

    result = pd.DataFrame(index=df.index)
    for ccy in currencies:
        result[f"FXL_{ccy}"] = np.nan
    if return_residuals:
        for col, _, _ in pairs:
            result[f"FXR_{col[3:]}"] = np.nan

    if not pairs:
        return result

    currency_idx = {ccy: i for i, ccy in enumerate(currencies)}
    base_a = np.zeros((len(pairs), len(currencies)), dtype=float)
    for i, (_, base, quote) in enumerate(pairs):
        base_a[i, currency_idx[base]] = 1.0
        base_a[i, currency_idx[quote]] = -1.0

    values = df[fx_cols].to_numpy(dtype=float)
    assert np.isfinite(values).all() and (values > 0).all()

    logs = np.log(values)

    # The extra row fixes the otherwise arbitrary common scale.
    constrained_a = np.vstack([base_a, np.ones(len(currencies))])
    constrained_y = np.vstack([logs.T, np.zeros(len(df))])
    log_values = np.linalg.lstsq(constrained_a, constrained_y, rcond=None)[0].T

    result.loc[:, [f"FXL_{ccy}" for ccy in currencies]] = np.exp(log_values)

    if return_residuals:
        predicted = log_values @ base_a.T
        residuals = logs - predicted
        result.loc[:, [f"FXR_{col[3:]}" for col, _, _ in pairs]] = residuals

    return result


def process_ohlc(
    df: pd.DataFrame,
    stem: str,
) -> pd.DataFrame:
    """Replace OHLC columns with derived features for one instrument."""
    O = f"{stem}_open"
    H = f"{stem}_high"
    L = f"{stem}_low"
    C = f"{stem}_close"

    if C not in df.columns:
        print(f"No close column provided for stem {stem}")
        relevant_cols = [col for col in df.columns if col.startswith(stem)]
        print(f"Relevant columns for stem {stem}: {relevant_cols}")
        return df

    assert C in df.columns

    ohl_cols = [O, H, L]
    has_ohl = [col in df.columns for col in ohl_cols]
    assert all(has_ohl) or not any(has_ohl)

    cols_to_drop = ohl_cols + [C] if all(has_ohl) else [C]
    close_col = df[C].astype(float)
    result = df.drop(columns=cols_to_drop).copy()
    if not any(has_ohl):
        print(f"Only close column provided for stem {stem}")
        result[f"{stem}_weighted_close"] = close_col
        return result

    vO = df[O].astype(float)
    vH = df[H].astype(float)
    vL = df[L].astype(float)
    vC = df[C].astype(float)

    result[f"{stem}_weighted_close"] = (vH + vL + 2.0 * vC) / 4.0
    RS = (vH - vC) * (vH - vO) + (vL - vC) * (vL - vO)
    result[f"{stem}_intraday_volatility"] = RS
    result[f"{stem}_overnight_return"] = (vO - vC.shift(1)).fillna(0.0)
    result[f"{stem}_day_return"] = vC - vO
    result[f"{stem}_close_in_range"] = (vC - vL) / (vH - vL).replace(0.0, np.nan)

    return result


def parse_groups(df: pd.DataFrame) -> dict[str, str]:
    relevant_cols = [
        col
        for col in df.select_dtypes(include=[np.floating]).columns
        if "fx" not in col.lower()
    ]
    print(f"Found {len(relevant_cols)} relevant columns")
    by_stem = defaultdict(set)
    for col in relevant_cols:
        stem, suffix = col.rsplit("_", 1)
        by_stem[stem].add(suffix)
    return by_stem


def zscore_robust(
    df: pd.DataFrame, W: int, clip_value: float | None = 3.0
) -> pd.DataFrame:
    assert W > 0
    float_cols = df.select_dtypes(include=[np.floating]).columns
    values = df[float_cols]
    assert np.isfinite(values.to_numpy()).all()
    center = values.rolling(window=W, min_periods=1).mean()
    scale = 1.25 * (values - center).abs().rolling(window=W, min_periods=1).mean()
    eps = 1e-12
    result = df.copy()
    result[float_cols] = (values - center) / (scale + eps)
    result = result.iloc[W:]
    assert np.isfinite(result[float_cols].to_numpy()).all()
    if clip_value is not None:
        result[float_cols] = result[float_cols].clip(
            lower=-clip_value, upper=clip_value
        )
    return result


if __name__ == "__main__":
    # test_df = pd.DataFrame(
    #     {
    #         "FX_EURUSD": [1.2],
    #         "FX_USDRUB": [100.0],
    #         "FX_EURRUB": [100.0],
    #     }
    # )
    # test_result = process_fx(test_df)
    # print(test_result.round(12))

    N = 1000
    core = np.random.randn(N)
    sigma = 1.0
    ols_coefs = pca_find(
        core + sigma * np.random.randn(N),
        0.707 * core + sigma * np.random.randn(N),
        0.5 * core + sigma * np.random.randn(N),
        0.2 * core + sigma * np.random.randn(N),
    )
    print(ols_coefs.round(12))
