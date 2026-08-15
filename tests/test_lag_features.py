from __future__ import annotations

import pandas as pd
import pytest
from features.compute import (
    DEFAULT_CLOSE_LAGS,
    _normalize_lags,
    compute_close_lag_features,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

PRICES = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]


@pytest.fixture()
def minimal_df() -> pd.DataFrame:
    """Ten-row single-ticker OHLCV-like frame with a DatetimeIndex."""
    return pd.DataFrame(
        {"Close": PRICES},
        index=pd.date_range(start="2024-01-01", periods=len(PRICES), freq="D"),
    )


@pytest.fixture()
def ohlcv_df(minimal_df) -> pd.DataFrame:
    """Minimal frame extended with dummy OHLV columns."""
    df = minimal_df.copy()
    df["Open"] = df["Close"] - 1
    df["High"] = df["Close"] + 2
    df["Low"] = df["Close"] - 2
    df["Volume"] = 1_000_000
    return df


# ---------------------------------------------------------------------------
# FR-FE-001: acceptance criteria 1 - Column Values
# ---------------------------------------------------------------------------


class TestLagColumnsPresent:
    def test_default_lags_create_four_columns(self, minimal_df):
        result = compute_close_lag_features(minimal_df)
        for lag in DEFAULT_CLOSE_LAGS:
            assert f"Close_lag_{lag}" in result.columns

    def test_custom_lags_produce_matching_columns(self, minimal_df):
        result = compute_close_lag_features(minimal_df, lags=[2, 7])
        assert "Close_lag_2" in result.columns
        assert "Close_lag_7" in result.columns
        assert "Close_lag_1" not in result.columns

    def test_original_columns_preserved(self, ohlcv_df):
        original_cols = list(ohlcv_df.columns)
        result = compute_close_lag_features(ohlcv_df)
        for col in original_cols:
            assert col in result.columns


# ---------------------------------------------------------------------------
# FR-FE-001: acceptance criteria 2 – Lag Correctness (Close_lag_k[t] == Close[t-k])
# ---------------------------------------------------------------------------


class TestLagCorrectness:
    def test_lag_1_values(self, minimal_df):
        result = compute_close_lag_features(minimal_df, lags=[1])
        for i in range(1, len(minimal_df)):
            expected = minimal_df["Close"].iloc[i - 1]
            actual = result["Close_lag_1"].iloc[i]
            assert actual == pytest.approx(
                expected
            ), f"Row {i}: Close_lag_1={actual}, expected {expected}"

    def test_lag_5_values(self, minimal_df):
        result = compute_close_lag_features(minimal_df, lags=[5])
        for i in range(5, len(minimal_df)):
            expected = minimal_df["Close"].iloc[i - 5]
            actual = result["Close_lag_5"].iloc[i]
            assert actual == pytest.approx(expected)

    def test_lag_10_values(self, minimal_df):
        """With 10 rows and lag=10 only row index 10+ would have a value;
        all rows in this fixture will be NaN confirms warmup behaviour."""
        result = compute_close_lag_features(minimal_df, lags=[10])
        assert result["Close_lag_10"].isna().all()

    def test_lag_20_values(self, minimal_df):
        result = compute_close_lag_features(minimal_df, lags=[20])
        assert result["Close_lag_20"].isna().all()

    def test_all_default_lags_correctness(self, minimal_df):
        result = compute_close_lag_features(minimal_df)
        for lag in (1, 5):
            for i in range(lag, len(minimal_df)):
                expected = minimal_df["Close"].iloc[i - lag]
                actual = result[f"Close_lag_{lag}"].iloc[i]
                assert actual == pytest.approx(expected)


# ---------------------------------------------------------------------------
# FR-FE-001: acceptance criteria 3 – no future values used (leakage safety)
# ---------------------------------------------------------------------------


class TestLeakageSafety:
    def test_lag_never_exceeds_current_close(self, minimal_df):
        """For a strictly ascending series, lag_k at row i must be ≤ Close[i]."""
        result = compute_close_lag_features(minimal_df, lags=[1, 5])
        for lag in (1, 5):
            col = f"Close_lag_{lag}"
            valid = result[col].dropna()
            aligned_close = result["Close"].loc[valid.index]
            assert (
                valid <= aligned_close
            ).all(), f"Lag {lag} produced a value greater than Close (possible leakage)"

    def test_lag_first_k_rows_are_null(self, minimal_df):
        """The first k rows must be NaN for lag_k (warmup window)."""
        for lag in (1, 5):
            result = compute_close_lag_features(minimal_df, lags=[lag])
            col = f"Close_lag_{lag}"
            assert (
                result[col].iloc[:lag].isna().all()
            ), f"Lag {lag}: expected first {lag} rows to be NaN"

    def test_input_dataframe_not_mutated(self, minimal_df):
        original_cols = list(minimal_df.columns)
        compute_close_lag_features(minimal_df)
        assert list(minimal_df.columns) == original_cols


# ---------------------------------------------------------------------------
# FR-FE-001: acceptance criteria 4 – original columns retained
# ---------------------------------------------------------------------------


class TestOutputShape:
    def test_row_count_unchanged_without_dropna(self, minimal_df):
        result = compute_close_lag_features(minimal_df, drop_na=False)
        assert len(result) == len(minimal_df)

    def test_row_count_reduced_with_dropna(self, minimal_df):
        """drop_na=True removes rows where any lag column is null (first 20)."""
        result = compute_close_lag_features(
            minimal_df, lags=DEFAULT_CLOSE_LAGS, drop_na=True
        )
        # With 10 rows and max lag=20, every row has at least one NaN → 0 rows remain.
        assert len(result) == 0

    def test_drop_na_with_small_lag(self, minimal_df):
        """With lag=1, only the first row is dropped."""
        result = compute_close_lag_features(minimal_df, lags=[1], drop_na=True)
        assert len(result) == len(minimal_df) - 1

    def test_index_preserved(self, minimal_df):
        result = compute_close_lag_features(minimal_df, drop_na=False)
        pd.testing.assert_index_equal(result.index, minimal_df.index)


# ---------------------------------------------------------------------------
# FR-FE-001: acceptance criteria 5 – clear errors on bad input
# ---------------------------------------------------------------------------


class TestValidation:
    def test_raises_on_empty_dataframe(self):
        empty = pd.DataFrame({"Close": pd.Series([], dtype=float)})
        with pytest.raises(ValueError, match="empty"):
            compute_close_lag_features(empty)

    def test_raises_when_close_column_missing(self, minimal_df):
        df_no_close = minimal_df.rename(columns={"Close": "close"})
        with pytest.raises(ValueError, match="Missing required column"):
            compute_close_lag_features(df_no_close)

    def test_raises_on_empty_lags(self, minimal_df):
        with pytest.raises(ValueError, match="At least one lag"):
            compute_close_lag_features(minimal_df, lags=[])

    def test_raises_on_zero_lag(self, minimal_df):
        with pytest.raises(ValueError, match="positive integers"):
            compute_close_lag_features(minimal_df, lags=[0])

    def test_raises_on_negative_lag(self, minimal_df):
        with pytest.raises(ValueError, match="positive integers"):
            compute_close_lag_features(minimal_df, lags=[-1])

    def test_raises_on_float_lag(self, minimal_df):
        with pytest.raises((ValueError, TypeError)):
            compute_close_lag_features(minimal_df, lags=[1.5])  # type: ignore[list-item]

    def test_raises_on_ambiguous_multi_ticker_close(self):
        """A multi-ticker frame where Close is a DataFrame should be rejected.

        yfinance returns a MultiIndex frame (price_type, ticker) when multiple
        tickers are requested. Selecting raw["Close"] from such a frame yields a
        DataFrame of per-ticker close prices rather than a Series. Passing that
        DataFrame directly to the feature function must be rejected.
        """
        idx = pd.date_range("2024-01-01", periods=5)
        # Simulate the result of yfinance_frame["Close"] for two tickers.
        close_df = pd.DataFrame(
            {"AAPL": [1, 2, 3, 4, 5], "MSFT": [6, 7, 8, 9, 10]},
            index=idx,
        )
        # Re-wrap so the outer frame has a "Close" column whose value is a DataFrame.
        outer = pd.concat({"Close": close_df}, axis=1)
        with pytest.raises(ValueError, match="ambiguous"):
            compute_close_lag_features(outer)


# ---------------------------------------------------------------------------
# _normalize_lags internal helper
# ---------------------------------------------------------------------------


class TestNormalizeLags:
    def test_deduplicates_and_sorts(self):
        assert _normalize_lags([5, 1, 5, 10]) == [1, 5, 10]

    def test_single_lag(self):
        assert _normalize_lags([3]) == [3]

    def test_raises_on_empty(self):
        with pytest.raises(ValueError):
            _normalize_lags([])
