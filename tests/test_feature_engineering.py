import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import (
    add_lag_features,
    add_rolling_features,
    add_calendar_features,
    add_cyclical_encodings,
    build_features,
)


@pytest.fixture
def simple_demand_series():
    """Create a simple demand series with known values for leakage testing."""
    dates = pd.date_range("2024-01-01", periods=10, freq="30min")
    df = pd.DataFrame({
        "Demand": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    }, index=dates)
    df.index.name = "DATE"
    return df


@pytest.fixture
def demand_series_for_rolling():
    """Create a demand series for rolling feature testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="30min")
    values = list(range(10, 110, 1))
    df = pd.DataFrame({"Demand": values}, index=dates)
    df.index.name = "DATE"
    return df


class TestLagFeatures:
    def test_lag_columns_created(self, simple_demand_series):
        """Test that all lag columns are created."""
        result = add_lag_features(simple_demand_series)

        expected_lags = ["Demand_lag1", "Demand_lag2", "Demand_lag48", "Demand_lag336"]
        for lag_col in expected_lags:
            assert lag_col in result.columns

    def test_lag1_correctness(self, simple_demand_series):
        """Test that Demand_lag1 at row t equals Demand at row t-1 (no leakage)."""
        result = add_lag_features(simple_demand_series)

        # Row 0 should be NaN (no previous value)
        assert pd.isna(result.iloc[0]["Demand_lag1"])

        # Row 1: lag1 should equal Demand from row 0
        assert result.iloc[1]["Demand_lag1"] == result.iloc[0]["Demand"]
        assert result.iloc[1]["Demand_lag1"] == 10.0

        # Row 2: lag1 should equal Demand from row 1
        assert result.iloc[2]["Demand_lag1"] == result.iloc[1]["Demand"]
        assert result.iloc[2]["Demand_lag1"] == 20.0

        # Row 5: lag1 should equal Demand from row 4
        assert result.iloc[5]["Demand_lag1"] == result.iloc[4]["Demand"]
        assert result.iloc[5]["Demand_lag1"] == 50.0

    def test_lag2_correctness(self, simple_demand_series):
        """Test that Demand_lag2 at row t equals Demand at row t-2."""
        result = add_lag_features(simple_demand_series)

        # First 2 rows should be NaN
        assert pd.isna(result.iloc[0]["Demand_lag2"])
        assert pd.isna(result.iloc[1]["Demand_lag2"])

        # Row 2: lag2 should equal Demand from row 0
        assert result.iloc[2]["Demand_lag2"] == result.iloc[0]["Demand"]
        assert result.iloc[2]["Demand_lag2"] == 10.0

    def test_lag48_correctness(self, demand_series_for_rolling):
        """Test that Demand_lag48 at row t equals Demand at row t-48."""
        result = add_lag_features(demand_series_for_rolling)

        # First 48 rows should have NaN for lag48
        assert result.iloc[:48]["Demand_lag48"].isna().all()

        # Row 48: lag48 should equal Demand from row 0
        assert result.iloc[48]["Demand_lag48"] == result.iloc[0]["Demand"]

        # Row 50: lag48 should equal Demand from row 2
        assert result.iloc[50]["Demand_lag48"] == result.iloc[2]["Demand"]

    def test_no_data_leakage_all_lags(self, simple_demand_series):
        """Comprehensive leakage test: all lags only use past values."""
        result = add_lag_features(simple_demand_series)

        for lag in [1, 2, 48, 336]:
            lag_col = f"Demand_lag{lag}"

            for t in range(lag, len(result)):
                # At time t, lag value should equal Demand at time t - lag
                expected = result.iloc[t - lag]["Demand"]
                actual = result.iloc[t][lag_col]
                assert actual == expected, f"Leakage detected at row {t} for {lag_col}"


class TestRollingFeatures:
    def test_rolling_columns_created(self, demand_series_for_rolling):
        """Test that all rolling feature columns are created."""
        result = add_rolling_features(demand_series_for_rolling)

        windows = [48, 336]
        stats = ['mean', 'std', 'min', 'max']

        for window in windows:
            for stat in stats:
                col_name = f'Demand_roll{window}_{stat}'
                assert col_name in result.columns

    def test_rolling_mean_correctness(self, demand_series_for_rolling):
        """Test that rolling mean only uses past values (no leakage)."""
        result = add_rolling_features(demand_series_for_rolling)

        # For window=48, first 48 rows should have NaN (need shift(1) first)
        assert result.iloc[:48]["Demand_roll48_mean"].isna().all()

        # Row 49: should be mean of rows 1-48 (after shift(1))
        # shift(1) means we're using demand from t-1, t-2, ..., t-48
        expected_values = result.iloc[1:49]["Demand"].values
        expected_mean = np.mean(expected_values)
        actual_mean = result.iloc[49]["Demand_roll48_mean"]

        assert np.isclose(actual_mean, expected_mean, rtol=1e-10)

    def test_rolling_std_correctness(self, demand_series_for_rolling):
        """Test that rolling std only uses past values."""
        result = add_rolling_features(demand_series_for_rolling)

        # Row 49: should be std of rows 1-48 (after shift(1))
        expected_values = result.iloc[1:49]["Demand"].values
        expected_std = np.std(expected_values, ddof=1)
        actual_std = result.iloc[49]["Demand_roll48_std"]

        assert np.isclose(actual_std, expected_std, rtol=1e-10)

    def test_rolling_min_correctness(self, demand_series_for_rolling):
        """Test that rolling min only uses past values."""
        result = add_rolling_features(demand_series_for_rolling)

        # Row 49: should be min of rows 1-48 (after shift(1))
        expected_min = result.iloc[1:49]["Demand"].min()
        actual_min = result.iloc[49]["Demand_roll48_min"]

        assert actual_min == expected_min

    def test_rolling_max_correctness(self, demand_series_for_rolling):
        """Test that rolling max only uses past values."""
        result = add_rolling_features(demand_series_for_rolling)

        # Row 49: should be max of rows 1-48 (after shift(1))
        expected_max = result.iloc[1:49]["Demand"].max()
        actual_max = result.iloc[49]["Demand_roll48_max"]

        assert actual_max == expected_max

    def test_no_current_value_in_rolling(self, demand_series_for_rolling):
        """Test that current row's demand is not included in rolling stats."""
        result = add_rolling_features(demand_series_for_rolling)

        # At row t, rolling mean should NOT include demand[t]
        for t in range(49, len(result)):
            # Get rolling mean at t
            rolling_mean = result.iloc[t]["Demand_roll48_mean"]

            # Manually compute: should use rows t-48 to t-1 (due to shift(1))
            start_idx = t - 48
            expected_mean = result.iloc[start_idx:t]["Demand"].mean()

            assert np.isclose(rolling_mean, expected_mean, rtol=1e-10)


class TestCalendarFeatures:
    def test_calendar_columns_created(self, simple_demand_series):
        """Test that all calendar feature columns are created."""
        result = add_calendar_features(simple_demand_series)

        expected_cols = [
            'hour', 'half_hour_of_day', 'day_of_week', 'day_of_month',
            'month', 'quarter', 'is_weekend', 'is_holiday'
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_hour_feature(self, simple_demand_series):
        """Test hour extraction from timestamp."""
        result = add_calendar_features(simple_demand_series)

        # 2024-01-01 00:00 should be hour 0
        assert result.iloc[0]["hour"] == 0
        # 30 minutes later (01:00) should be hour 1
        assert result.iloc[1]["hour"] == 0
        assert result.iloc[2]["hour"] == 1

    def test_half_hour_of_day_feature(self, simple_demand_series):
        """Test half-hour-of-day feature (0-47 for 48 slots in a day)."""
        result = add_calendar_features(simple_demand_series)

        # 00:00 should be half_hour_of_day 0
        assert result.iloc[0]["half_hour_of_day"] == 0
        # 00:30 should be half_hour_of_day 1
        assert result.iloc[1]["half_hour_of_day"] == 1
        # 01:00 should be half_hour_of_day 2
        assert result.iloc[2]["half_hour_of_day"] == 2

    def test_day_of_week_feature(self, simple_demand_series):
        """Test day of week feature."""
        result = add_calendar_features(simple_demand_series)

        # 2024-01-01 is a Monday (0)
        assert result.iloc[0]["day_of_week"] == 0

    def test_is_weekend_feature(self, simple_demand_series):
        """Test is_weekend feature."""
        result = add_calendar_features(simple_demand_series)

        # 2024-01-01 is Monday, not weekend
        assert result.iloc[0]["is_weekend"] == 0

        # Add a test case for Saturday (2024-01-06)
        dates = pd.date_range("2024-01-06", periods=2, freq="30min")
        df = pd.DataFrame({"Demand": [100.0, 101.0]}, index=dates)
        df.index.name = "DATE"
        result = add_calendar_features(df)

        # 2024-01-06 is Saturday (5)
        assert result.iloc[0]["day_of_week"] == 5
        assert result.iloc[0]["is_weekend"] == 1

    def test_month_feature(self, simple_demand_series):
        """Test month feature."""
        result = add_calendar_features(simple_demand_series)

        # 2024-01-01 is January (month 1)
        assert result.iloc[0]["month"] == 1

    def test_quarter_feature(self, simple_demand_series):
        """Test quarter feature."""
        result = add_calendar_features(simple_demand_series)

        # January is Q1
        assert result.iloc[0]["quarter"] == 1

    def test_is_holiday_peru(self):
        """Test is_holiday feature for Peruvian holidays (Fiestas Patrias)."""
        # Create synthetic series covering July 28, 2024 (Fiestas Patrias) and July 15, 2024
        dates = pd.date_range("2024-07-15", periods=28, freq="30min")
        df = pd.DataFrame({"Demand": range(len(dates))}, index=dates)
        df.index.name = "DATE"
        result = add_calendar_features(df)

        # July 15, 2024 is not a holiday
        july_15_indices = result.index.date == pd.Timestamp("2024-07-15").date()
        assert (result[july_15_indices]["is_holiday"] == 0).all()

        # July 28, 2024 is Fiestas Patrias (Peru's Independence Day)
        july_28_indices = result.index.date == pd.Timestamp("2024-07-28").date()
        assert (result[july_28_indices]["is_holiday"] == 1).all()


class TestCyclicalEncodings:
    def test_cyclical_columns_created(self, simple_demand_series):
        """Test that all cyclical encoding columns are created."""
        # Must call add_calendar_features first to create the required columns
        df = add_calendar_features(simple_demand_series)
        result = add_cyclical_encodings(df)

        expected_cols = [
            'hour_sin', 'hour_cos',
            'day_of_week_sin', 'day_of_week_cos',
            'month_sin', 'month_cos',
            'half_hour_of_day_sin', 'half_hour_of_day_cos',
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_cyclical_encoding_bounds(self, simple_demand_series):
        """Test that sin/cos values are within [-1, 1]."""
        df = add_calendar_features(simple_demand_series)
        result = add_cyclical_encodings(df)

        sin_cols = [c for c in result.columns if c.endswith('_sin')]
        cos_cols = [c for c in result.columns if c.endswith('_cos')]

        for col in sin_cols + cos_cols:
            assert (result[col] >= -1.0).all()
            assert (result[col] <= 1.0).all()

    def test_hour_encoding_periodicity(self):
        """Test that hour encoding at hour 0 and hour 24 are similar."""
        dates1 = pd.date_range("2024-01-01 00:00", periods=1, freq="30min")
        df1 = pd.DataFrame({"Demand": [100.0]}, index=dates1)
        df1.index.name = "DATE"
        result1 = add_cyclical_encodings(add_calendar_features(df1))

        dates2 = pd.date_range("2024-01-02 00:00", periods=1, freq="30min")
        df2 = pd.DataFrame({"Demand": [100.0]}, index=dates2)
        df2.index.name = "DATE"
        result2 = add_cyclical_encodings(add_calendar_features(df2))

        # Both should have the same hour encoding (hour 0)
        assert np.isclose(result1.iloc[0]["hour_sin"], result2.iloc[0]["hour_sin"])
        assert np.isclose(result1.iloc[0]["hour_cos"], result2.iloc[0]["hour_cos"])


class TestBuildFeatures:
    def test_build_features_includes_all_columns(self, demand_series_for_rolling):
        """Test that build_features includes all expected feature columns."""
        result = build_features(demand_series_for_rolling, drop_na=False)

        # Original column
        assert "Demand" in result.columns

        # Lag features
        assert "Demand_lag1" in result.columns
        assert "Demand_lag48" in result.columns

        # Rolling features
        assert "Demand_roll48_mean" in result.columns
        assert "Demand_roll336_std" in result.columns

        # Calendar features
        assert "hour" in result.columns
        assert "is_weekend" in result.columns

        # Cyclical encodings
        assert "hour_sin" in result.columns
        assert "month_cos" in result.columns

    def test_build_features_drop_na_false(self, demand_series_for_rolling):
        """Test build_features with drop_na=False keeps NaN rows."""
        result = build_features(demand_series_for_rolling, drop_na=False)

        # Should have NaN values from lags and rolling features
        assert result["Demand_lag336"].isna().sum() > 0

    def test_build_features_drop_na_true(self, demand_series_for_rolling):
        """Test build_features with drop_na=True removes NaN rows."""
        result = build_features(demand_series_for_rolling, drop_na=True)

        # Should have no NaN values
        assert result.isna().sum().sum() == 0

        # Should have fewer rows than original
        assert len(result) < len(demand_series_for_rolling)

    def test_build_features_preserves_index_name(self, demand_series_for_rolling):
        """Test that DATE index name is preserved through feature building."""
        result = build_features(demand_series_for_rolling, drop_na=False)

        assert result.index.name == "DATE"
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_feature_engineering_no_leakage(self, demand_series_for_rolling):
        """Test that full feature engineering pipeline doesn't introduce leakage."""
        result = build_features(demand_series_for_rolling, drop_na=False)

        # At row t, Demand_lag1 should equal original Demand at row t-1
        for t in range(1, 50):  # Check first 50 rows
            if not pd.isna(result.iloc[t]["Demand_lag1"]):
                assert result.iloc[t]["Demand_lag1"] == result.iloc[t - 1]["Demand"]

        # At row t, rolling mean should use values prior to t
        for t in range(49, 100):
            rolling_mean = result.iloc[t]["Demand_roll48_mean"]
            start_idx = t - 48
            expected_mean = result.iloc[start_idx:t]["Demand"].mean()

            if not pd.isna(rolling_mean):
                assert np.isclose(rolling_mean, expected_mean, rtol=1e-10)

    def test_build_features_output_shape(self, demand_series_for_rolling):
        """Test that build_features output has correct number of rows and columns."""
        result = build_features(demand_series_for_rolling, drop_na=True)

        # Should have same index as input
        assert isinstance(result, pd.DataFrame)

        # Should have many more columns due to features
        assert result.shape[1] > demand_series_for_rolling.shape[1]

        # Should have fewer rows due to drop_na=True
        assert result.shape[0] < demand_series_for_rolling.shape[0]
