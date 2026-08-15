from __future__ import annotations

import math

import pandas as pd
import pytest
from features.compute import (
    compute_atr,
    compute_bollinger_bands,
    compute_macd,
    compute_rsi,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N = 50  # enough rows to clear all warmup windows


@pytest.fixture()
def close_df() -> pd.DataFrame:
    """50-row single-ticker frame with a DatetimeIndex and a synthetic close series."""
    prices = [100 + i * 0.5 + (i % 5) * 2 for i in range(N)]
    return pd.DataFrame(
        {"Close": prices},
        index=pd.date_range(start="2024-01-01", periods=N, freq="D"),
    )


@pytest.fixture()
def ohlcv_df(close_df) -> pd.DataFrame:
    """close_df extended with Open / High / Low / Volume columns."""
    df = close_df.copy()
    df["Open"] = df["Close"] - 0.5
    df["High"] = df["Close"] + 1.0
    df["Low"] = df["Close"] - 1.0
    df["Volume"] = 1_000_000
    return df


@pytest.fixture()
def empty_df() -> pd.DataFrame:
    return pd.DataFrame({"Close": [], "High": [], "Low": []})


# ---------------------------------------------------------------------------
# FR-FE-002: compute_rsi
# ---------------------------------------------------------------------------


class TestComputeRsiOutputShape:
    def test_rsi_column_added_with_default_window(self, close_df):
        result = compute_rsi(close_df)
        assert "rsi_14" in result.columns

    def test_rsi_column_added_with_custom_window(self, close_df):
        result = compute_rsi(close_df, window=7)
        assert "rsi_7" in result.columns

    def test_original_columns_preserved(self, close_df):
        result = compute_rsi(close_df)
        assert "Close" in result.columns

    def test_row_count_unchanged(self, close_df):
        result = compute_rsi(close_df)
        assert len(result) == len(close_df)

    def test_input_dataframe_not_mutated(self, close_df):
        original_cols = list(close_df.columns)
        compute_rsi(close_df)
        assert list(close_df.columns) == original_cols


class TestComputeRsiWarmup:
    def test_first_window_rows_are_nan(self, close_df, window=14):
        result = compute_rsi(close_df, window=window)
        assert result[f"rsi_{window}"].iloc[:window].isna().all()

    def test_values_present_after_warmup(self, close_df, window=14):
        result = compute_rsi(close_df, window=window)
        assert result[f"rsi_{window}"].iloc[window:].notna().all()


class TestComputeRsiValues:
    def test_rsi_range_0_to_100(self, close_df):
        result = compute_rsi(close_df)
        valid = result["rsi_14"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_is_100_on_all_gains(self):
        """When price only rises, RSI should be 100 after warmup."""
        prices = [100 + i for i in range(30)]
        df = pd.DataFrame(
            {"Close": prices},
            index=pd.date_range("2024-01-01", periods=30, freq="D"),
        )
        result = compute_rsi(df, window=5)
        valid = result["rsi_5"].dropna()
        assert valid.values == pytest.approx(100.0)

    def test_rsi_is_0_on_all_losses(self):
        """When price only falls, RSI should be 0 after warmup."""
        prices = [100 - i for i in range(30)]
        df = pd.DataFrame(
            {"Close": prices},
            index=pd.date_range("2024-01-01", periods=30, freq="D"),
        )
        result = compute_rsi(df, window=5)
        valid = result["rsi_5"].dropna()
        assert valid.values == pytest.approx(0.0)


class TestComputeRsiValidation:
    def test_raises_on_empty_dataframe(self, empty_df):
        with pytest.raises(ValueError, match="empty"):
            compute_rsi(empty_df)

    def test_raises_on_missing_close_column(self, close_df):
        df = close_df.rename(columns={"Close": "close"})
        with pytest.raises(ValueError, match="Missing required column"):
            compute_rsi(df)

    def test_raises_on_zero_window(self, close_df):
        with pytest.raises(ValueError, match="positive"):
            compute_rsi(close_df, window=0)

    def test_raises_on_negative_window(self, close_df):
        with pytest.raises(ValueError, match="positive"):
            compute_rsi(close_df, window=-1)

    def test_custom_close_column_name(self, close_df):
        df = close_df.rename(columns={"Close": "Adj Close"})
        result = compute_rsi(df, close_column="Adj Close")
        assert "rsi_14" in result.columns


# ---------------------------------------------------------------------------
# FR-FE-002: compute_macd
# ---------------------------------------------------------------------------


class TestComputeMacdOutputShape:
    def test_three_columns_added(self, close_df):
        result = compute_macd(close_df)
        for col in ("macd_line", "macd_signal", "macd_histogram"):
            assert col in result.columns

    def test_original_columns_preserved(self, close_df):
        result = compute_macd(close_df)
        assert "Close" in result.columns

    def test_row_count_unchanged(self, close_df):
        result = compute_macd(close_df)
        assert len(result) == len(close_df)

    def test_input_dataframe_not_mutated(self, close_df):
        original_cols = list(close_df.columns)
        compute_macd(close_df)
        assert list(close_df.columns) == original_cols


class TestComputeMacdWarmup:
    def test_macd_line_nan_during_slow_warmup(self, close_df):
        """macd_line is NaN for the first slow-1 rows."""
        result = compute_macd(close_df, fast=12, slow=26, signal=9)
        assert result["macd_line"].iloc[:25].isna().all()

    def test_signal_nan_during_full_warmup(self, close_df):
        """macd_signal is NaN for the first slow+signal-2 rows."""
        result = compute_macd(close_df, fast=12, slow=26, signal=9)
        warmup = 25 + 8  # slow-1 + signal-1
        assert result["macd_signal"].iloc[:warmup].isna().all()


class TestComputeMacdValues:
    def test_histogram_equals_line_minus_signal(self, close_df):
        result = compute_macd(close_df)
        valid = result.dropna(subset=["macd_line", "macd_signal", "macd_histogram"])
        expected = valid["macd_line"] - valid["macd_signal"]
        pd.testing.assert_series_equal(
            valid["macd_histogram"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )


class TestComputeMacdValidation:
    def test_raises_on_empty_dataframe(self, empty_df):
        with pytest.raises(ValueError, match="empty"):
            compute_macd(empty_df)

    def test_raises_on_missing_close_column(self, close_df):
        df = close_df.rename(columns={"Close": "close"})
        with pytest.raises(ValueError, match="Missing required column"):
            compute_macd(df)

    def test_raises_when_fast_equals_slow(self, close_df):
        with pytest.raises(ValueError, match="fast window must be smaller"):
            compute_macd(close_df, fast=12, slow=12)

    def test_raises_when_fast_greater_than_slow(self, close_df):
        with pytest.raises(ValueError, match="fast window must be smaller"):
            compute_macd(close_df, fast=26, slow=12)

    def test_raises_on_zero_fast(self, close_df):
        with pytest.raises(ValueError, match="positive"):
            compute_macd(close_df, fast=0, slow=26)

    def test_raises_on_zero_signal(self, close_df):
        with pytest.raises(ValueError, match="positive"):
            compute_macd(close_df, fast=12, slow=26, signal=0)

    def test_custom_close_column_name(self, close_df):
        df = close_df.rename(columns={"Close": "Adj Close"})
        result = compute_macd(df, close_column="Adj Close")
        assert "macd_line" in result.columns


# ---------------------------------------------------------------------------
# FR-FE-002: compute_bollinger_bands
# ---------------------------------------------------------------------------


class TestComputeBollingerBandsOutputShape:
    def test_four_columns_added(self, close_df):
        result = compute_bollinger_bands(close_df)
        for col in ("bb_middle", "bb_upper", "bb_lower", "bb_bandwidth"):
            assert col in result.columns

    def test_original_columns_preserved(self, close_df):
        result = compute_bollinger_bands(close_df)
        assert "Close" in result.columns

    def test_row_count_unchanged(self, close_df):
        result = compute_bollinger_bands(close_df)
        assert len(result) == len(close_df)

    def test_input_dataframe_not_mutated(self, close_df):
        original_cols = list(close_df.columns)
        compute_bollinger_bands(close_df)
        assert list(close_df.columns) == original_cols


class TestComputeBollingerBandsWarmup:
    def test_first_window_minus_one_rows_are_nan(self, close_df):
        result = compute_bollinger_bands(close_df, window=20)
        assert result["bb_middle"].iloc[:19].isna().all()

    def test_values_present_after_warmup(self, close_df):
        result = compute_bollinger_bands(close_df, window=20)
        assert result["bb_middle"].iloc[19:].notna().all()


class TestComputeBollingerBandsValues:
    def test_upper_band_above_middle(self, close_df):
        result = compute_bollinger_bands(close_df).dropna(subset=["bb_middle"])
        assert (result["bb_upper"] >= result["bb_middle"]).all()

    def test_lower_band_below_middle(self, close_df):
        result = compute_bollinger_bands(close_df).dropna(subset=["bb_middle"])
        assert (result["bb_lower"] <= result["bb_middle"]).all()

    def test_bandwidth_formula(self, close_df):
        result = compute_bollinger_bands(close_df).dropna(subset=["bb_middle"])
        expected = (result["bb_upper"] - result["bb_lower"]) / result["bb_middle"]
        pd.testing.assert_series_equal(
            result["bb_bandwidth"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_bandwidth_is_non_negative(self, close_df):
        result = compute_bollinger_bands(close_df).dropna(subset=["bb_bandwidth"])
        assert (result["bb_bandwidth"] >= 0).all()

    def test_middle_band_equals_rolling_mean(self, close_df):
        window = 20
        result = compute_bollinger_bands(close_df, window=window)
        expected = close_df["Close"].rolling(window=window, min_periods=window).mean()
        pd.testing.assert_series_equal(result["bb_middle"], expected, check_names=False)

    @pytest.mark.parametrize("num_std", [1.0, 2.0, 3.0])
    def test_band_width_scales_with_num_std(self, close_df, num_std):
        result = compute_bollinger_bands(close_df, num_std=num_std).dropna(
            subset=["bb_middle"]
        )
        std = close_df["Close"].rolling(20, min_periods=20).std(ddof=1).dropna()
        expected_width = 2 * num_std * std
        actual_width = (result["bb_upper"] - result["bb_lower"]).dropna()
        pd.testing.assert_series_equal(
            actual_width.reset_index(drop=True),
            expected_width.reset_index(drop=True),
            check_names=False,
        )


class TestComputeBollingerBandsValidation:
    def test_raises_on_empty_dataframe(self, empty_df):
        with pytest.raises(ValueError, match="empty"):
            compute_bollinger_bands(empty_df)

    def test_raises_on_missing_close_column(self, close_df):
        df = close_df.rename(columns={"Close": "close"})
        with pytest.raises(ValueError, match="Missing required column"):
            compute_bollinger_bands(df)

    def test_raises_on_zero_window(self, close_df):
        with pytest.raises(ValueError, match="positive"):
            compute_bollinger_bands(close_df, window=0)

    def test_raises_on_negative_window(self, close_df):
        with pytest.raises(ValueError, match="positive"):
            compute_bollinger_bands(close_df, window=-5)

    def test_custom_close_column_name(self, close_df):
        df = close_df.rename(columns={"Close": "Adj Close"})
        result = compute_bollinger_bands(df, close_column="Adj Close")
        assert "bb_middle" in result.columns


# ---------------------------------------------------------------------------
# FR-FE-002: compute_atr
# ---------------------------------------------------------------------------


class TestComputeAtrOutputShape:
    def test_two_columns_added(self, ohlcv_df):
        result = compute_atr(ohlcv_df)
        assert "true_range" in result.columns
        assert "atr_14" in result.columns

    def test_custom_window_column_name(self, ohlcv_df):
        result = compute_atr(ohlcv_df, window=7)
        assert "atr_7" in result.columns

    def test_original_columns_preserved(self, ohlcv_df):
        result = compute_atr(ohlcv_df)
        for col in ("Close", "High", "Low"):
            assert col in result.columns

    def test_row_count_unchanged(self, ohlcv_df):
        result = compute_atr(ohlcv_df)
        assert len(result) == len(ohlcv_df)

    def test_input_dataframe_not_mutated(self, ohlcv_df):
        original_cols = list(ohlcv_df.columns)
        compute_atr(ohlcv_df)
        assert list(ohlcv_df.columns) == original_cols


class TestComputeAtrWarmup:
    def test_first_row_true_range_falls_back_to_high_minus_low(self, ohlcv_df):
        """First row lacks prev_close, so TR = max(H-L, NaN, NaN) = H-L."""
        result = compute_atr(ohlcv_df, window=14)
        expected = ohlcv_df["High"].iloc[0] - ohlcv_df["Low"].iloc[0]
        assert result["true_range"].iloc[0] == pytest.approx(expected)

    def test_first_window_atr_rows_are_nan(self, ohlcv_df):
        """min_periods=14 means 13 leading NaNs (rows 0-12); row 13 is the first value."""
        result = compute_atr(ohlcv_df, window=14)
        assert result["atr_14"].iloc[:13].isna().all()

    def test_atr_values_present_after_warmup(self, ohlcv_df):
        result = compute_atr(ohlcv_df, window=14)
        assert result["atr_14"].iloc[14:].notna().all()


class TestComputeAtrValues:
    def test_true_range_is_non_negative(self, ohlcv_df):
        result = compute_atr(ohlcv_df)
        assert (result["true_range"].dropna() >= 0).all()

    def test_atr_is_non_negative(self, ohlcv_df):
        result = compute_atr(ohlcv_df)
        assert (result["atr_14"].dropna() >= 0).all()

    def test_true_range_at_least_high_minus_low(self, ohlcv_df):
        """TR >= High - Low by definition."""
        result = compute_atr(ohlcv_df)
        valid = result.dropna(subset=["true_range"])
        hl = valid["High"] - valid["Low"]
        assert (valid["true_range"] >= hl - 1e-9).all()

    def test_true_range_computed_correctly(self):
        """Manual verification for a two-row dataset with a known gap."""
        df = pd.DataFrame(
            {
                "Close": [100.0, 105.0],
                "High": [102.0, 110.0],
                "Low": [98.0, 103.0],
            },
            index=pd.date_range("2024-01-01", periods=2, freq="D"),
        )
        result = compute_atr(df, window=1)
        # Row 1: prev_close=100, high=110, low=103
        # H-L=7, |H-prev|=10, |L-prev|=3 → TR=10
        assert result["true_range"].iloc[1] == pytest.approx(10.0)


class TestComputeAtrValidation:
    def test_raises_on_empty_dataframe(self, empty_df):
        with pytest.raises(ValueError, match="empty"):
            compute_atr(empty_df)

    def test_raises_on_missing_high_column(self, close_df):
        with pytest.raises(ValueError, match="Missing required columns"):
            compute_atr(close_df)  # no High / Low

    def test_raises_on_missing_single_column(self, ohlcv_df):
        df = ohlcv_df.drop(columns=["High"])
        with pytest.raises(ValueError, match="Missing required columns"):
            compute_atr(df)

    def test_raises_on_zero_window(self, ohlcv_df):
        with pytest.raises(ValueError, match="positive"):
            compute_atr(ohlcv_df, window=0)

    def test_raises_on_negative_window(self, ohlcv_df):
        with pytest.raises(ValueError, match="positive"):
            compute_atr(ohlcv_df, window=-3)

    def test_custom_column_names(self, ohlcv_df):
        df = ohlcv_df.rename(columns={"High": "H", "Low": "L", "Close": "C"})
        result = compute_atr(df, high_column="H", low_column="L", close_column="C")
        assert "atr_14" in result.columns
