from __future__ import annotations

import time

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


ASSETS = [
    "US_Stock_ACWI_adj_close",
    "US_Stock_BND_adj_close",
    "US_Stock_EMB_adj_close",
    "US_Stock_VALE_adj_close",
    "US_Stock_VEA_adj_close",
    "US_Stock_XLE_adj_close",
]

CACHE_DIR = Path("data/yfinance_cache")
TRAIN_PATH = Path("data/train.csv")
MAPPING_PATH = Path("data/date_id_mapping.csv")
DATED_TRAIN_PATH = Path("data/dated_train.csv")
START_DATE = "2017-01-01"
DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_SLEEP_SECONDS = 5.0


@dataclass(frozen=True)
class Source:
    ticker: str
    multiplier: float = 1.0
    adjusted: bool = True


SOURCES = {
    "US_Stock_ACWI_adj_close": [Source("ACWI")],
    "US_Stock_BND_adj_close": [Source("BND")],
    "US_Stock_EMB_adj_close": [Source("EMB")],
    "US_Stock_VALE_adj_close": [Source("VALE")],
    "US_Stock_VEA_adj_close": [Source("VEA")],
    "US_Stock_XLE_adj_close": [Source("XLE")],
}


def _cache_path(source: Source) -> Path:
    safe_ticker = source.ticker.replace("=", "_").replace("/", "_")
    safe_multiplier = f"{source.multiplier:.12g}".replace(".", "p")
    adjusted = "adj" if source.adjusted else "close"
    safe_start = START_DATE.replace("-", "")
    return CACHE_DIR / f"{safe_ticker}_{safe_multiplier}_{adjusted}_{safe_start}.csv"


def _read_cached_close(source: Source) -> pd.Series:
    path = _cache_path(source)
    if not path.exists():
        return pd.Series(dtype=float)

    cached = pd.read_csv(path, index_col=0, parse_dates=True)
    close = cached["close"].dropna().astype(float)
    close.name = source.ticker
    return close


def _write_cached_close(source: Source, close: pd.Series) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    close.rename("close").to_csv(_cache_path(source))


def _download_close(source: Source) -> pd.Series:
    cached = _read_cached_close(source)
    if not cached.empty:
        return cached

    for attempt in range(DOWNLOAD_ATTEMPTS):
        data = yf.download(
            source.ticker,
            start=START_DATE,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if not data.empty:
            break

        if attempt + 1 < DOWNLOAD_ATTEMPTS:
            time.sleep(DOWNLOAD_SLEEP_SECONDS)
    else:
        return pd.Series(dtype=float)

    price_col = "Adj Close" if source.adjusted and "Adj Close" in data else "Close"
    close = data[price_col]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = close.dropna().astype(float) * source.multiplier
    close.name = source.ticker
    _write_cached_close(source, close)
    return close


def download_asset(asset: str) -> pd.Series:
    errors = []
    for source in SOURCES[asset]:
        try:
            close = _download_close(source)
        except Exception as exc:  # yfinance raises several transport errors.
            errors.append(f"{source.ticker}: {exc}")
            continue

        if not close.empty:
            print(f"{asset}: using {source.ticker}")
            return close.rename(asset)

        errors.append(f"{source.ticker}: no close data")

    msg = "\n".join(errors)
    raise RuntimeError(f"Could not download {asset} from yfinance:\n{msg}")


def build_prices() -> pd.DataFrame:
    prices = pd.concat([download_asset(asset) for asset in ASSETS], axis=1)
    return prices.dropna().sort_index()


def load_train_prices() -> pd.DataFrame:
    train = pd.read_csv(TRAIN_PATH, usecols=["date_id", *ASSETS])
    return train.sort_values("date_id").reset_index(drop=True)


def save_dated_train(mapping: pd.DataFrame) -> None:
    train = pd.read_csv(TRAIN_PATH)
    dated_train = train.merge(mapping, on="date_id", how="left", validate="one_to_one")
    assert dated_train["date"].notna().all()

    dates = pd.to_datetime(dated_train["date"])
    dated_train["date"] = dates.dt.strftime("%Y-%m-%d")
    dated_train["dom"] = dates.dt.day
    dated_train["dow"] = dates.dt.dayofweek + 1

    calendar_cols = ["date", "dom", "dow"]
    ordered_cols = ["date_id", *calendar_cols]
    ordered_cols.extend(col for col in dated_train.columns if col not in ordered_cols)
    dated_train = dated_train.loc[:, ordered_cols]
    dated_train.to_csv(DATED_TRAIN_PATH, index=False)


def infer_date_mapping(train: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    yahoo_dates = set(prices.index.normalize())
    dates = []
    current = pd.Timestamp("2018-01-01")

    for is_us_holiday in train[ASSETS].isna().all(axis=1):
        while True:
            is_weekday = current.weekday() < 5
            is_yahoo_open = current.normalize() in yahoo_dates
            if is_weekday and (is_yahoo_open != is_us_holiday):
                dates.append(current)
                current += pd.Timedelta(days=1)
                break
            current += pd.Timedelta(days=1)

    return pd.DataFrame({"date_id": train["date_id"], "date": dates})


def estimate_multipliers(
    train: pd.DataFrame,
    prices: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    aligned_yahoo = prices.reindex(pd.DatetimeIndex(mapping["date"]))
    aligned_yahoo.index = train.index

    rows = []
    for col in ASSETS:
        valid = train[col].notna() & aligned_yahoo[col].notna()
        log_ratio = np.log(train.loc[valid, col]) - np.log(
            aligned_yahoo.loc[valid, col]
        )
        rows.append(
            {
                "asset": col,
                "ticker": SOURCES[col][0].ticker,
                "multiplier_mean": np.exp(log_ratio.mean()),
                "multiplier_median": np.exp(log_ratio.median()),
                "log_ratio_std": log_ratio.std(),
                "n": valid.sum(),
            }
        )

    return pd.DataFrame(rows)


def return_alignment_mse(
    train: pd.DataFrame,
    prices: pd.DataFrame,
    mapping: pd.DataFrame,
) -> float:
    aligned_yahoo = prices.reindex(pd.DatetimeIndex(mapping["date"]))
    aligned_yahoo.index = train.index

    errors = []
    for col in ASSETS:
        valid = train[col].notna() & aligned_yahoo[col].notna()
        train_returns = np.diff(np.log(train.loc[valid, col].to_numpy(float)))
        yahoo_returns = np.diff(np.log(aligned_yahoo.loc[valid, col].to_numpy(float)))
        errors.append((train_returns - yahoo_returns) ** 2)

    return float(np.mean(np.concatenate(errors)))


def main() -> None:
    train = load_train_prices()
    prices = build_prices()
    if prices.empty:
        raise RuntimeError("No overlapping dates across downloaded assets.")

    mapping = infer_date_mapping(train, prices)
    multipliers = estimate_multipliers(train, prices, mapping)
    return_mse = return_alignment_mse(train, prices, mapping)
    mapping.to_csv(MAPPING_PATH, index=False)
    save_dated_train(mapping)

    print(
        f"\nInferred date range: {mapping.date.iloc[0].date()}"
        f" -> {mapping.date.iloc[-1].date()}"
    )
    print(f"Saved full mapping to {MAPPING_PATH}")
    print(f"Saved dated train to {DATED_TRAIN_PATH}")
    print("\nSample correspondence:")
    sample = pd.concat([mapping.head(12), mapping.tail(5)])
    print(sample.to_string(index=False))
    print(f"\nUS-stock return alignment MSE: {return_mse:.10f}")
    print("\nEstimated train / current-yfinance-adjusted multipliers:")
    print(multipliers.round(8).to_string(index=False))


if __name__ == "__main__":
    main()
