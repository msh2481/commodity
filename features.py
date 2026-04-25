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
    X_train = (
        pd.read_csv("data/dated_train.csv").drop(columns=["date"]).set_index("date_id")
    )
    X_train = common_preprocessing(X_train, W=W)
    X_primary, X_aux = primary_aux_split(X_train)
    X_input = make_input_features(X_train)

    Y_train = pd.read_csv("data/train_labels.csv")
    Y_train = Y_train.ffill().bfill()

    qt = QuantileTransformer(output_distribution="normal", random_state=42)
    Y_train = pd.DataFrame(
        qt.fit_transform(Y_train), columns=Y_train.columns, index=Y_train.index
    )
    Y_train = Y_train.iloc[W:]

    assert len(set(len(a) for a in [X_input, X_primary, X_aux, Y_train])) == 1

    return X_input, X_primary, X_aux, Y_train


def _write_feature_values(path: str, values: pd.Series) -> None:
    values = values.sort_values(ascending=False)
    with open(path, "w") as f:
        for feature, value in values.items():
            f.write(f"{feature} {value:.4f}\n")


def _ridge_coefs(X: np.ndarray, Y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    y_mean = Y.mean(axis=0)
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - y_mean

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    shrinkage = S / (S**2 + alpha)
    feature_coefs = Vt.T @ (shrinkage[:, None] * (U.T @ Y_centered))
    intercept = y_mean - X.mean(axis=0) @ feature_coefs
    return np.vstack([intercept, feature_coefs])


def _ridge_predict(X: np.ndarray, coefs: np.ndarray) -> np.ndarray:
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    return X_with_intercept @ coefs


def _mse(Y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean((Y - pred) ** 2))


def _r2_by_target(Y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    ss_res = ((Y - pred) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / ss_tot


def _rowwise_spearman(Y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.array(
        [
            spearmanr(target_row, pred_row).statistic
            for target_row, pred_row in zip(Y, pred)
        ]
    )


def _print_spearman_summary(label: str, row_spearman: np.ndarray) -> None:
    row_spearman = row_spearman[np.isfinite(row_spearman)]
    if len(row_spearman) == 0:
        print(f"{label} spearman: mean=nan std=nan sharpe=nan")
        return
    mean = float(row_spearman.mean())
    std = float(row_spearman.std())
    sharpe = mean / std if std > 0.0 else np.nan
    print(f"{label} spearman: mean={mean:.4f} std={std:.4f} sharpe={sharpe:.4f}")


def _permutation_importance_mse(
    X: np.ndarray,
    Y: np.ndarray,
    coefs: np.ndarray,
    n_repeats: int = 1,
) -> np.ndarray:
    rng = np.random.default_rng(0)
    baseline = _mse(Y, _ridge_predict(X, coefs))
    importances = np.zeros(X.shape[1])

    for col_idx in range(X.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            X_permuted[:, col_idx] = rng.permutation(X_permuted[:, col_idx])
            scores.append(_mse(Y, _ridge_predict(X_permuted, coefs)) - baseline)
        importances[col_idx] = np.mean(scores)

    return importances


def _fit_and_score_features(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    label: str,
) -> tuple[pd.Series, pd.Series]:
    X_values = X.to_numpy(dtype=float)
    Y_values = Y.to_numpy(dtype=float)
    split = len(X_values) // 2
    assert 0 < split < len(X_values)

    X_train, X_test = X_values[:split], X_values[split:]
    Y_train, Y_test = Y_values[:split], Y_values[split:]

    coefs = _ridge_coefs(X_train, Y_train, alpha=1.0)
    pred = _ridge_predict(X_test, coefs)
    r2 = pd.Series(
        _r2_by_target(Y_test, pred),
        index=Y.columns,
        name="r2",
    )

    importance = pd.Series(
        _permutation_importance_mse(X_test, Y_test, coefs),
        index=X.columns,
        name="feature_importance",
    )

    print(f"{label} mean r2: {r2.mean():.4f}")
    _print_spearman_summary(label, _rowwise_spearman(Y_test, pred))
    return r2, importance


def check_features() -> None:
    X_input, X_primary, X_aux, Y_train = get_datasets()
    X = X_input.select_dtypes(include=[np.number]).iloc[:-1]
    X_primary = X_primary.select_dtypes(include=[np.number]).iloc[1:]
    X_aux = X_aux.select_dtypes(include=[np.number]).iloc[1:]
    Y_train = Y_train.select_dtypes(include=[np.number]).iloc[:-1]
    print("Datasets ready")
    r2_primary, feature_imp_primary = _fit_and_score_features(X, X_primary, "Primary")
    print("Primary done")
    r2_aux, feature_imp_aux = _fit_and_score_features(X, X_aux, "Aux")
    print("Aux done")
    r2_y, feature_imp_y = _fit_and_score_features(X, Y_train, "Y")
    print("Y done")
    _write_feature_values("r2_primary.txt", r2_primary)
    _write_feature_values("r2_aux.txt", r2_aux)
    _write_feature_values("r2_y.txt", r2_y)
    _write_feature_values("feature_imp_primary.txt", feature_imp_primary)
    _write_feature_values("feature_imp_aux.txt", feature_imp_aux)
    _write_feature_values("feature_imp_y.txt", feature_imp_y)
    print("Written to files")


if __name__ == "__main__":
    check_features()
