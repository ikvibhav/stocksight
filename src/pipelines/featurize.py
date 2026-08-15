from pathlib import Path

import pandas as pd
import yaml
from pipelines.ingest import (
    EXPECTED_COLUMNS,
    build_filename,
    fetch_data,
    save_data,
    validate_data,
)
from prefect import flow, task
from features.compute import (
    compute_atr,
    compute_bollinger_bands,
    compute_calendar_features,
    compute_close_lag_features,
    compute_macd,
    compute_rsi,
)

DEFAULT_FEATURES_CONFIG = Path("configs/features_config.yaml")


def _load_features_config(config_path: Path) -> dict:
    """Load and return the feature toggles config from a YAML file."""
    with config_path.open() as fh:
        return yaml.safe_load(fh)


# Implements FR-FE-005
RAW_PATH = Path("data/raw/")
RAW_PATH.mkdir(parents=True, exist_ok=True)
PROCESSED_PATH = Path("data/processed/")
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)


def _build_filename(ticker: str, period: str, suffix: str = "") -> str:
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    stem = f"{ticker}_{period}_{timestamp}"
    if suffix:
        stem = f"{stem}_{suffix}"
    return f"{stem}.csv"


def _coerce_single_ticker_frame(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Extract a single ticker OHLCV frame from yfinance output."""

    # 1. Multi-ticker downloads return MultiIndex columns: (field, ticker).
    if isinstance(data.columns, pd.MultiIndex):
        level_1 = set(data.columns.get_level_values(1))
        if ticker not in level_1:
            raise ValueError(
                "Ticker "
                f"{ticker} not found in downloaded data columns: "
                f"{sorted(level_1)}"
            )
        data = data.xs(ticker, axis=1, level=1)

    # 2. Check expected columns are present
    missing = [c for c in EXPECTED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError("Missing expected columns after flattening: " f"{missing}")

    return data


@task
def build_feature_dataframe(
    data: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Apply feature groups driven by configs/features_config.yaml.

    Each group is only computed when its ``enabled`` flag is true.
    Parameters (windows, lags, etc.) are read from config.
    Implements: FR-FE-004
    """
    featured = data.copy()

    if config.get("calendar_features", {}).get("enabled", False):
        featured = compute_calendar_features(featured)

    lag_cfg = config.get("lag_features", {})
    if lag_cfg.get("enabled", False):
        featured = compute_close_lag_features(
            featured, lags=lag_cfg["lags"], drop_na=False
        )

    rsi_cfg = config.get("rsi", {})
    if rsi_cfg.get("enabled", False):
        featured = compute_rsi(featured, window=rsi_cfg["window"])

    macd_cfg = config.get("macd", {})
    if macd_cfg.get("enabled", False):
        featured = compute_macd(
            featured,
            fast=macd_cfg["fast"],
            slow=macd_cfg["slow"],
            signal=macd_cfg["signal"],
        )

    bb_cfg = config.get("bollinger_bands", {})
    if bb_cfg.get("enabled", False):
        featured = compute_bollinger_bands(
            featured,
            window=bb_cfg["window"],
            num_std=bb_cfg["num_std"],
        )

    atr_cfg = config.get("atr", {})
    if atr_cfg.get("enabled", False):
        featured = compute_atr(featured, window=atr_cfg["window"])

    if config.get("output", {}).get("drop_na", False):
        featured = featured.dropna()

    return featured


@task
def save_processed_data(data: pd.DataFrame, tickers: list[str], period: str) -> Path:
    ticker_label = "_".join(tickers)
    filename = _build_filename(ticker=ticker_label, period=period, suffix="features")
    output = PROCESSED_PATH / filename
    data.to_csv(output, index=True)
    print(f"Processed feature data saved: {output}")
    return output


@flow
def feature_engineering_pipeline(
    tickers: list[str],
    period: str = "1y",
    save_raw: bool = True,
    save_processed: bool = True,
    config_path: Path = DEFAULT_FEATURES_CONFIG,
) -> pd.DataFrame:
    """
    End-to-end ingestion + feature generation pipeline.

    Steps:
    1) Download OHLCV data for one or more tickers.
    2) Validate schema once on the downloaded frame.
    3) Split per ticker and compute all feature sets per ticker.
    4) Concatenate into one dataframe and optionally save outputs.
    """
    if not tickers:
        raise ValueError("tickers must contain at least one symbol.")

    downloaded = fetch_data(ticker=tickers, period=period)
    if not validate_data(downloaded):
        raise ValueError("Data ingestion validation failed.")

    if save_raw:
        save_data(downloaded, build_filename(tickers, period))

    per_ticker_frames: list[pd.DataFrame] = []
    for symbol in tickers:
        ticker_data = _coerce_single_ticker_frame(downloaded, ticker=symbol)
        featured = build_feature_dataframe(
            ticker_data, config=_load_features_config(config_path)
        )
        featured = featured.copy()
        featured["Ticker"] = symbol
        per_ticker_frames.append(featured)

    combined = pd.concat(per_ticker_frames, axis=0).sort_index()

    if save_processed:
        save_processed_data(combined, tickers=tickers, period=period)

    return combined


if __name__ == "__main__":
    feature_engineering_pipeline(tickers=["AAPL", "MSFT"], period="1y")
