import numpy as np
import pandas as pd


def process_fx(df: pd.DataFrame, residuals: bool = False) -> pd.DataFrame:
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
    if residuals:
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
    valid = np.isfinite(values) & (values > 0)

    for row_i, index in enumerate(df.index):
        mask = valid[row_i]
        if not mask.any():
            continue

        a = base_a[mask]
        y = np.log(values[row_i, mask])

        # The extra row fixes the otherwise arbitrary common scale.
        constrained_a = np.vstack([a, np.ones(len(currencies))])
        constrained_y = np.concatenate([y, [0.0]])
        log_values, *_ = np.linalg.lstsq(constrained_a, constrained_y, rcond=None)

        result.loc[index, [f"FXL_{ccy}" for ccy in currencies]] = log_values

        predicted = base_a @ log_values
        observed = np.full(len(pairs), np.nan, dtype=float)
        observed[mask] = np.log(values[row_i, mask])
        residuals = observed - predicted
        if residuals:
            for pair_i, (col, _, _) in enumerate(pairs):
                result.loc[index, f"FXR_{col[3:]}"] = residuals[pair_i]

    return result


if __name__ == "__main__":
    test_df = pd.DataFrame(
        {
            "FX_EURUSD": [1.2],
            "FX_USDRUB": [100.0],
            "FX_EURRUB": [100.0],
        }
    )
    test_result = process_fx(test_df)

    print(test_result.round(12))
    assert np.allclose(
        test_result[["FXL_EUR", "FXL_RUB", "FXL_USD"]].sum(axis=1),
        0.0,
    )
