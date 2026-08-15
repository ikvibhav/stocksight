from __future__ import annotations

import pandas as pd
import pytest
from features.compute import compute_calendar_features

# ---------------------------------------------------------------------------
# Expected calendar feature columns
# ---------------------------------------------------------------------------

CALENDAR_COLUMNS = [
    "day_of_week",
    "day_of_month",
    "day_of_year",
    "month",
    "quarter",
    "is_month_start",
    "is_month_end",
    "is_quarter_start",
    "is_quarter_end",
    "is_year_start",
    "is_year_end",
]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def daily_df() -> pd.DataFrame:
    """One full year of daily data (2024) with a DatetimeIndex and a dummy Close."""
    idx = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
    return pd.DataFrame({"Close": range(len(idx))}, index=idx)


@pytest.fixture()
def single_row_df() -> pd.DataFrame:
    """Single-row frame on a known date: 2024-01-01 (Monday, year start)."""
    return pd.DataFrame(
        {"Close": [100.0]},
        index=pd.DatetimeIndex(["2024-01-01"]),
    )


@pytest.fixture()
def boundary_df() -> pd.DataFrame:
    """
    Frame containing key boundary dates for easy assertion:
      2024-01-01  – year start, quarter start, month start (Monday)
      2024-01-31  – month end
      2024-03-31  – quarter end, month end
      2024-04-01  – quarter start, month start
      2024-12-31  – year end, quarter end, month end (Tuesday)
    """
    dates = ["2024-01-01", "2024-01-31", "2024-03-31", "2024-04-01", "2024-12-31"]
    return pd.DataFrame(
        {"Close": [1.0, 2.0, 3.0, 4.0, 5.0]},
        index=pd.DatetimeIndex(dates),
    )


# ---------------------------------------------------------------------------
# FR-FE-003: Output shape
# ---------------------------------------------------------------------------


class TestComputeCalendarFeaturesOutputShape:
    def test_all_eleven_columns_added(self, daily_df):
        result = compute_calendar_features(daily_df)
        for col in CALENDAR_COLUMNS:
            assert col in result.columns

    def test_original_columns_preserved(self, daily_df):
        result = compute_calendar_features(daily_df)
        assert "Close" in result.columns

    def test_row_count_unchanged(self, daily_df):
        result = compute_calendar_features(daily_df)
        assert len(result) == len(daily_df)

    def test_input_dataframe_not_mutated(self, daily_df):
        original_cols = list(daily_df.columns)
        compute_calendar_features(daily_df)
        assert list(daily_df.columns) == original_cols

    def test_no_nulls_introduced(self, daily_df):
        result = compute_calendar_features(daily_df)
        for col in CALENDAR_COLUMNS:
            assert result[col].notna().all(), f"NaN found in column {col}"


# ---------------------------------------------------------------------------
# FR-FE-003: Correctness – numeric calendar fields
# ---------------------------------------------------------------------------


class TestComputeCalendarFeaturesNumericValues:
    def test_day_of_week_for_known_monday(self, boundary_df):
        """2024-01-01 is a Monday → dayofweek == 0."""
        result = compute_calendar_features(boundary_df)
        assert result.loc["2024-01-01", "day_of_week"] == 0

    def test_day_of_week_for_known_tuesday(self, boundary_df):
        """2024-12-31 is a Tuesday → dayofweek == 1."""
        result = compute_calendar_features(boundary_df)
        assert result.loc["2024-12-31", "day_of_week"] == 1

    def test_day_of_month_values(self, boundary_df):
        result = compute_calendar_features(boundary_df)
        assert result.loc["2024-01-01", "day_of_month"] == 1
        assert result.loc["2024-01-31", "day_of_month"] == 31

    def test_day_of_year_jan_1(self, boundary_df):
        result = compute_calendar_features(boundary_df)
        assert result.loc["2024-01-01", "day_of_year"] == 1

    def test_day_of_year_dec_31_leap_year(self, boundary_df):
        """2024 is a leap year → Dec 31 is day 366."""
        result = compute_calendar_features(boundary_df)
        assert result.loc["2024-12-31", "day_of_year"] == 366

    def test_month_values(self, boundary_df):
        result = compute_calendar_features(boundary_df)
        assert result.loc["2024-01-01", "month"] == 1
        assert result.loc["2024-03-31", "month"] == 3
        assert result.loc["2024-04-01", "month"] == 4
        assert result.loc["2024-12-31", "month"] == 12

    @pytest.mark.parametrize(
        "date,expected_quarter",
        [
            ("2024-01-01", 1),
            ("2024-03-31", 1),
            ("2024-04-01", 2),
            ("2024-12-31", 4),
        ],
    )
    def test_quarter_assignment(self, boundary_df, date, expected_quarter):
        result = compute_calendar_features(boundary_df)
        assert result.loc[date, "quarter"] == expected_quarter

    def test_full_year_day_of_week_range(self, daily_df):
        result = compute_calendar_features(daily_df)
        assert result["day_of_week"].between(0, 6).all()

    def test_full_year_month_range(self, daily_df):
        result = compute_calendar_features(daily_df)
        assert result["month"].between(1, 12).all()

    def test_full_year_quarter_range(self, daily_df):
        result = compute_calendar_features(daily_df)
        assert result["quarter"].between(1, 4).all()


# ---------------------------------------------------------------------------
# FR-FE-003: Correctness – boolean boundary flags
# ---------------------------------------------------------------------------


class TestComputeCalendarFeaturesBooleanFlags:
    def test_is_month_start_on_first_of_month(self, boundary_df):
        result = compute_calendar_features(boundary_df)
        assert result.loc["2024-01-01", "is_month_start"] == True
        assert result.loc["2024-04-01", "is_month_start"] == True

    def test_is_month_start_false_on_non_first(self, boundary_df):
        result = compute_calendar_features(boundary_df)
        assert result.loc["2024-01-31", "is_month_start"] == False

    def test_is_month_end_on_last_of_month(self, boundary_df):
        result = compute_calendar_features(boundary_df)
        assert result.loc["2024-01-31", "is_month_end"] == True
        assert result.loc["2024-03-31", "is_month_end"] == True
        assert result.loc["2024-12-31", "is_month_end"] == True

    def test_is_month_end_false_on_non_last(self, boundary_df):
        result = compute_calendar_features(boundary_df)
        assert result.loc["2024-01-01", "is_month_end"] == False

    def test_is_quarter_start_on_quarter_boundary(self, boundary_df):
        result = compute_calendar_features(boundary_df)
        assert result.loc["2024-01-01", "is_quarter_start"] == True
        assert result.loc["2024-04-01", "is_quarter_start"] == True

    def test_is_quarter_start_false_on_non_boundary(self, boundary_df):
        result = compute_calendar_features(boundary_df)
        assert result.loc["2024-01-31", "is_quarter_start"] == False

    def test_is_quarter_end_on_quarter_boundary(self, boundary_df):
        result = compute_calendar_features(boundary_df)
        assert result.loc["2024-03-31", "is_quarter_end"] == True
        assert result.loc["2024-12-31", "is_quarter_end"] == True

    def test_is_quarter_end_false_on_non_boundary(self, boundary_df):
        result = compute_calendar_features(boundary_df)
        assert result.loc["2024-01-01", "is_quarter_end"] == False

    def test_is_year_start_on_jan_1(self, boundary_df):
        result = compute_calendar_features(boundary_df)
        assert result.loc["2024-01-01", "is_year_start"] == True

    def test_is_year_start_false_on_other_dates(self, boundary_df):
        result = compute_calendar_features(boundary_df)
        assert result.loc["2024-01-31", "is_year_start"] == False
        assert result.loc["2024-12-31", "is_year_start"] == False

    def test_is_year_end_on_dec_31(self, boundary_df):
        result = compute_calendar_features(boundary_df)
        assert result.loc["2024-12-31", "is_year_end"] == True

    def test_is_year_end_false_on_other_dates(self, boundary_df):
        result = compute_calendar_features(boundary_df)
        assert result.loc["2024-01-01", "is_year_end"] == False
        assert result.loc["2024-03-31", "is_year_end"] == False

    def test_full_year_has_twelve_month_starts(self, daily_df):
        result = compute_calendar_features(daily_df)
        assert result["is_month_start"].sum() == 12

    def test_full_year_has_twelve_month_ends(self, daily_df):
        result = compute_calendar_features(daily_df)
        assert result["is_month_end"].sum() == 12

    def test_full_year_has_four_quarter_starts(self, daily_df):
        result = compute_calendar_features(daily_df)
        assert result["is_quarter_start"].sum() == 4

    def test_full_year_has_four_quarter_ends(self, daily_df):
        result = compute_calendar_features(daily_df)
        assert result["is_quarter_end"].sum() == 4

    def test_full_year_has_one_year_start(self, daily_df):
        result = compute_calendar_features(daily_df)
        assert result["is_year_start"].sum() == 1

    def test_full_year_has_one_year_end(self, daily_df):
        result = compute_calendar_features(daily_df)
        assert result["is_year_end"].sum() == 1


# ---------------------------------------------------------------------------
# FR-FE-003: Single-row edge case
# ---------------------------------------------------------------------------


class TestComputeCalendarFeaturesSingleRow:
    def test_single_row_returns_one_row(self, single_row_df):
        result = compute_calendar_features(single_row_df)
        assert len(result) == 1

    def test_single_row_all_columns_present(self, single_row_df):
        result = compute_calendar_features(single_row_df)
        for col in CALENDAR_COLUMNS:
            assert col in result.columns

    def test_single_row_correct_values_for_jan_1(self, single_row_df):
        result = compute_calendar_features(single_row_df)
        row = result.iloc[0]
        assert row["day_of_week"] == 0  # Monday
        assert row["day_of_month"] == 1
        assert row["day_of_year"] == 1
        assert row["month"] == 1
        assert row["quarter"] == 1
        assert row["is_month_start"] == True
        assert row["is_quarter_start"] == True
        assert row["is_year_start"] == True
        assert row["is_year_end"] == False


# ---------------------------------------------------------------------------
# FR-FE-003: Validation
# ---------------------------------------------------------------------------


class TestComputeCalendarFeaturesValidation:
    def test_raises_on_empty_dataframe(self):
        df = pd.DataFrame({"Close": []}, index=pd.DatetimeIndex([]))
        with pytest.raises(ValueError, match="empty"):
            compute_calendar_features(df)

    def test_raises_on_integer_index(self):
        df = pd.DataFrame({"Close": [1, 2, 3]})
        with pytest.raises(ValueError, match="DatetimeIndex"):
            compute_calendar_features(df)

    def test_raises_on_string_index(self):
        df = pd.DataFrame(
            {"Close": [1, 2, 3]},
            index=["2024-01-01", "2024-01-02", "2024-01-03"],
        )
        with pytest.raises(ValueError, match="DatetimeIndex"):
            compute_calendar_features(df)

    def test_works_with_extra_columns(self):
        """Extra columns in the input must all be preserved."""
        df = pd.DataFrame(
            {"Close": [100.0], "Volume": [1_000_000]},
            index=pd.DatetimeIndex(["2024-06-15"]),
        )
        result = compute_calendar_features(df)
        assert "Volume" in result.columns
        assert "Close" in result.columns
