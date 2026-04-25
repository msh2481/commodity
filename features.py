f"""
Overall pipeline
================

Three sets of features:
1. Input features -- here we add everything that might help, engineered for convenience to the model
2. Main targets -- subset of features that are directly relevant to predicting spread, perhaps (H + L + 2C)/4 for each asset involved in target pairs
3. Aux targets -- being able to predict these would be a good sign, but not immediately relevant to predicting spreads; can include stuff like volatility, or assets not participating in spreads

All these share most of the preprocessing steps:
1. Drop US_Stock_GOLD_*, since its mostly nan; also drop `*_settlement_price` for simplicity, since they are almost equal to `*_Close`
2. Ffill and bfill to fill in missing values
3. Process FX columns using `process_fx` from `utils.py` (going from 38 to 9 cols)
4. log(max(., 1e-9)) everything (except date_id, date, dom and dow)
5. OHLC -> weighted_close = (H + L + 2C) / 4, intraday_volatility = (H - C) * (H - O) + (L - C) * (L - O), overnight_return = (O_t - C_(t-1)), day_return = (C_t - O_t), close_in_range = (C - L) / (H - L)
6. To make things stationary: using rolling window of W=90 days, compute robust z-scores as (x - mu) / sigma, where mu = IQM, sigma = (mean absolute deviation from MAD)

Then, for input features, on top of that, EMA-smooth some of the features, and add some basic indicators, like RSI or Stochastic RSI (based on weighted_close)
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import QuantileTransformer
from utils import parse_groups, process_fx, process_ohlc, zscore_robust


def common_preprocessing(df: pd.DataFrame, W: int = 90) -> pd.DataFrame:
    cols_to_drop = list(df.filter(regex=r"^US_Stock_GOLD_").columns) + list(
        df.filter(regex=r"^.*_settlement_price").columns
    )
    print("Dropping columns: ", cols_to_drop)
    df = df.drop(columns=cols_to_drop)
    df.columns = [
        col.lower().replace("open_interest", "openinterest") for col in df.columns
    ]
    df = df.ffill().bfill()

    fx_df = df.filter(regex=r"^FX_")
    new_fx_df = process_fx(fx_df)
    df = df.drop(columns=fx_df.columns)
    df = pd.concat([df, new_fx_df], axis=1)

    float_cols = df.select_dtypes(include=[np.floating]).columns
    df[float_cols] = np.log(df[float_cols].clip(lower=1e-9))

    results = [df[["dom", "dow"]].copy()]
    for stem, suffixes in parse_groups(df).items():
        if "close" not in suffixes:
            print(f"Skipping stem {stem} because it doesn't have a close column")
            continue
        selected = df[[f"{stem}_{suffix}" for suffix in suffixes]]
        result = process_ohlc(selected, stem)
        results.append(result)
    df = pd.concat(results, axis=1)
    df = zscore_robust(df, W=W)

    return df


def primary_aux_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = df.filter(like="weighted_close")
    # Auxiliary columns: all columns except the primary "weighted_close" columns and the core identifier columns
    core_cols = ["dom", "dow"]
    aux = df.drop(columns=primary.columns.tolist() + core_cols)
    return primary, aux


def make_input_features(df: pd.DataFrame) -> pd.DataFrame:
    return df


def get_datasets(W: int = 90):
    X_orig = (
        pd.read_csv("data/dated_train.csv").drop(columns=["date"]).set_index("date_id")
    )
    X_train = common_preprocessing(X_orig, W=W)
    X_primary, X_aux = primary_aux_split(X_train)
    X_input = make_input_features(X_train)

    Y_train = pd.read_csv("data/train_labels.csv")
    Y_train = Y_train.ffill().bfill().iloc[W:]

    assert len(set(len(a) for a in [X_input, X_primary, X_aux, Y_train])) == 1

    return X_input, X_primary, X_aux, X_orig.iloc[W:], Y_train
