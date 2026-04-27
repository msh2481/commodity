"""Simulate statistical power of metrics under controlled prediction signal.

The target and prediction-noise matrices are T x D log-returns with zero mean
and unit sample variance per column. Both use the supplied spectrum and share
the same asset covariance basis. Prediction noise is post-processed to have
heavy-tailed Student-t marginals.

Predictions are generated with true correlation c:
    pred = c * target + sqrt(1 - c**2) * noise
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, t as student_t


T_DEFAULT = 2_000
N_RUNS_DEFAULT = 20
SEED_DEFAULT = 42
C_VALUES_DEFAULT = np.linspace(0.0, 0.02, 9)
NOISE_DF_DEFAULT = 3.0

SINGULAR_VALUES = np.array(
    [
        273.95,
        186.00,
        125.72,
        106.29,
        86.21,
        70.85,
        64.37,
        51.57,
        48.97,
        48.22,
        45.78,
        43.81,
        42.79,
        41.58,
        41.35,
        39.26,
        37.78,
        36.84,
        35.96,
        34.58,
        33.21,
        32.95,
        32.34,
        31.77,
        31.12,
        30.48,
        29.55,
        29.44,
        28.29,
        28.15,
        27.16,
        26.68,
        25.97,
        25.56,
        25.23,
        24.60,
        24.33,
        24.16,
        23.99,
        23.66,
        23.39,
        23.04,
        22.58,
        22.35,
        21.97,
        21.82,
        21.54,
        21.30,
        20.81,
        20.52,
        20.19,
        19.84,
        19.80,
        19.66,
        19.23,
        18.83,
        18.51,
        18.48,
        18.30,
        17.82,
        17.47,
        17.19,
        17.01,
        16.84,
        16.69,
        16.20,
        15.83,
        15.27,
        15.07,
        14.64,
        14.38,
        14.28,
        13.55,
        13.13,
        12.59,
        11.52,
        11.20,
        10.49,
        10.02,
        9.58,
        8.72,
        8.35,
        7.75,
        7.49,
        6.63,
        6.44,
        6.24,
        6.02,
        5.63,
        4.61,
        4.09,
        3.98,
        3.83,
        3.21,
        3.15,
        2.93,
        2.69,
        2.52,
        2.39,
        2.09,
        1.93,
        1.81,
        1.56,
    ],
    dtype=float,
)


@dataclass(frozen=True)
class SpectrumDiagnostics:
    min_column_var: float
    max_column_var: float
    mean_abs_singular_error: float
    max_abs_singular_error: float
    achieved_singular_values: np.ndarray


def center_and_standardize_columns(x: np.ndarray) -> np.ndarray:
    """Return a copy with zero column mean and unit sample variance."""
    x = x - x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, ddof=1, keepdims=True)
    if np.any(std == 0):
        raise ValueError("Cannot standardize a column with zero variance.")
    return x / std


def random_orthonormal(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    """Sample a rows x cols matrix with orthonormal columns."""
    q, r = np.linalg.qr(rng.normal(size=(rows, cols)), mode="reduced")
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return q * signs


def sample_iid_returns(
    rng: np.random.Generator,
    n_rows: int,
    n_cols: int,
) -> np.ndarray:
    return center_and_standardize_columns(rng.normal(size=(n_rows, n_cols)))


def sample_spectrum_returns(
    rng: np.random.Generator,
    n_rows: int,
    singular_values: np.ndarray,
    right_basis: np.ndarray | None = None,
) -> np.ndarray:
    n_cols = len(singular_values)
    u = random_orthonormal(rng, n_rows, n_cols)
    v = random_orthonormal(rng, n_cols, n_cols) if right_basis is None else right_basis
    x = (u * singular_values) @ v.T
    return center_and_standardize_columns(x)


def make_student_t_marginals(x: np.ndarray, df: float) -> np.ndarray:
    if df <= 2:
        raise ValueError("Student-t df must be greater than 2 for finite variance.")
    x = center_and_standardize_columns(x)
    uniforms = np.clip(norm.cdf(x), 1e-12, 1 - 1e-12)
    return center_and_standardize_columns(student_t.ppf(uniforms, df=df))


def sample_predictions(
    target: np.ndarray,
    noise: np.ndarray,
    c: float,
) -> np.ndarray:
    if not 0 <= c <= 1:
        raise ValueError(f"Expected c in [0, 1], got {c}.")
    if target.shape != noise.shape:
        raise ValueError(
            f"Expected target/noise shapes to match, got {target.shape} and {noise.shape}."
        )
    return c * target + np.sqrt(1 - c**2) * noise


def sample_spectrum_target_noise_pair(
    rng: np.random.Generator,
    n_rows: int,
    singular_values: np.ndarray,
    noise_df: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_cols = len(singular_values)
    right_basis = random_orthonormal(rng, n_cols, n_cols)
    target = sample_spectrum_returns(rng, n_rows, singular_values, right_basis)
    gaussian_noise = sample_spectrum_returns(rng, n_rows, singular_values, right_basis)
    return (
        target,
        make_student_t_marginals(gaussian_noise, df=noise_df),
    )


def rowwise_spearman(target: np.ndarray, pred: np.ndarray) -> np.ndarray:
    target_ranks = rankdata(target, axis=1)
    pred_ranks = rankdata(pred, axis=1)
    return rowwise_pearson(target_ranks, pred_ranks)


def rowwise_gaussian_rank_correlation(
    target: np.ndarray, pred: np.ndarray
) -> np.ndarray:
    n_cols = target.shape[1]
    target_scores = norm.ppf((rankdata(target, axis=1) - 0.5) / n_cols)
    pred_scores = norm.ppf((rankdata(pred, axis=1) - 0.5) / n_cols)
    return rowwise_pearson(target_scores, pred_scores)


def rowwise_pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)

    numerator = np.sum(x * y, axis=1)
    denominator = np.sqrt(np.sum(x**2, axis=1) * np.sum(y**2, axis=1))
    return numerator / denominator


def compute_metrics(target: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    row_spearman = rowwise_spearman(target, pred)
    row_gcor = rowwise_gaussian_rank_correlation(target, pred)
    return {
        "neg_mse": -float(np.mean((pred - target) ** 2)),
        "spearman_mean": float(row_spearman.mean()),
        "gcor_mean": float(row_gcor.mean()),
    }


def spectrum_diagnostics(
    rng: np.random.Generator,
    n_rows: int,
    singular_values: np.ndarray,
) -> SpectrumDiagnostics:
    x = sample_spectrum_returns(rng, n_rows, singular_values)
    achieved = np.linalg.svd(x, compute_uv=False)
    column_vars = x.var(axis=0, ddof=1)
    abs_error = np.abs(achieved - singular_values)
    return SpectrumDiagnostics(
        min_column_var=float(column_vars.min()),
        max_column_var=float(column_vars.max()),
        mean_abs_singular_error=float(abs_error.mean()),
        max_abs_singular_error=float(abs_error.max()),
        achieved_singular_values=achieved,
    )


def run_simulation(
    rng: np.random.Generator,
    n_rows: int,
    singular_values: np.ndarray,
    c_values: np.ndarray,
    n_runs: int,
    noise_df: float,
) -> pd.DataFrame:
    rows = []

    for c in c_values:
        for run_idx in range(n_runs):
            target, noise = sample_spectrum_target_noise_pair(
                rng, n_rows, singular_values, noise_df
            )
            pred = sample_predictions(target, noise, float(c))
            for metric, value in compute_metrics(target, pred).items():
                rows.append(
                    {
                        "case": "spectrum_target_spectrum_t_noise",
                        "c": float(c),
                        "run": run_idx,
                        "metric": metric,
                        "value": value,
                    }
                )

    return pd.DataFrame(rows)


def summarize_power(results: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for (case_name, metric), group in results.groupby(["case", "metric"]):
        summary_rows.append(
            {
                "case": case_name,
                "metric": metric,
                "pearson_corr_c_value": group["c"].corr(
                    group["value"], method="pearson"
                ),
            }
        )
    return pd.DataFrame(summary_rows).sort_values(["case", "metric"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test metric power under controlled target-prediction correlation."
    )
    parser.add_argument("--n-runs", type=int, default=N_RUNS_DEFAULT)
    parser.add_argument("--t", type=int, default=T_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--noise-df", type=float, default=NOISE_DF_DEFAULT)
    parser.add_argument(
        "--c-values",
        type=float,
        nargs="+",
        default=C_VALUES_DEFAULT,
        help="True correlations to test, e.g. --c-values 0 .02 .05 .1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    c_values = np.asarray(args.c_values, dtype=float)

    diagnostics = spectrum_diagnostics(rng, args.t, SINGULAR_VALUES)
    results = run_simulation(
        rng=rng,
        n_rows=args.t,
        singular_values=SINGULAR_VALUES,
        c_values=c_values,
        n_runs=args.n_runs,
        noise_df=args.noise_df,
    )

    print(
        f"T={args.t}, D={len(SINGULAR_VALUES)}, n_runs={args.n_runs}, "
        f"noise_df={args.noise_df:g}"
    )
    print("\nSpectrum diagnostics after column standardization:")
    print(
        pd.Series(
            {
                "min_column_var": diagnostics.min_column_var,
                "max_column_var": diagnostics.max_column_var,
                "mean_abs_singular_error": diagnostics.mean_abs_singular_error,
                "max_abs_singular_error": diagnostics.max_abs_singular_error,
            }
        ).to_string(float_format="{:.6f}".format)
    )
    print("\nAchieved singular values after column standardization:")
    print(
        np.array2string(
            diagnostics.achieved_singular_values,
            precision=2,
            suppress_small=True,
            max_line_width=100,
        )
    )

    print("\nPearson correlation between true c and metric value:")
    print(summarize_power(results).to_string(index=False, float_format="{:.4f}".format))

    print("\nMean metric value by c:")
    mean_by_c = results.pivot_table(
        index=["case", "metric"],
        columns="c",
        values="value",
        aggfunc="mean",
    )
    print(mean_by_c.to_string(float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
