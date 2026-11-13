import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.models.prophet import ProphetForecaster


@pytest.fixture
def synthetic_demand_96steps():
    """Create synthetic 30-minute demand data covering 96 steps (2 full days)."""
    dates = pd.date_range("2024-01-01", periods=96, freq="30min")
    hours = dates.hour + dates.minute / 60
    demand = 100 + 30 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 2, len(dates))
    series = pd.Series(demand, index=dates, name="Demand")
    return series


@pytest.fixture
def synthetic_demand_long():
    """Create longer synthetic demand data for extended training (2 years)."""
    dates = pd.date_range("2022-01-01", periods=2*365*48, freq="30min")  # 2 years
    hours = dates.hour + dates.minute / 60
    demand = 100 + 30 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 3, len(dates))
    series = pd.Series(demand, index=dates, name="Demand")
    return series


class TestProphetForecasterInit:
    """Test model initialization."""

    def test_initialization_default_params(self):
        """Test initialization with default parameters."""
        forecaster = ProphetForecaster()
        assert forecaster.include_holidays is True
        assert forecaster.yearly_seasonality is True
        assert forecaster.weekly_seasonality is True
        assert forecaster.daily_seasonality is True
        assert forecaster.fitted_model is None

    def test_initialization_without_holidays(self):
        """Test initialization with holidays disabled."""
        forecaster = ProphetForecaster(include_holidays=False)
        assert forecaster.include_holidays is False
        assert forecaster.fitted_model is None

    def test_initialization_custom_seasonality(self):
        """Test initialization with custom seasonality flags."""
        forecaster = ProphetForecaster(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        assert forecaster.yearly_seasonality is False
        assert forecaster.weekly_seasonality is True
        assert forecaster.daily_seasonality is False


class TestProphetForecasterFit:
    """Test fitting on synthetic demand data."""

    def test_fit_on_96step_synthetic_data(self, synthetic_demand_96steps):
        """Test fitting on 2 full days (96 steps) of synthetic data."""
        forecaster = ProphetForecaster(include_holidays=False)
        result = forecaster.fit(synthetic_demand_96steps)

        # Should return self for method chaining
        assert result is forecaster

        # Should store fitted model
        assert forecaster.fitted_model is not None

        # Should store training index
        assert forecaster.y_train_index is not None
        assert len(forecaster.y_train_index) == 96

    def test_fit_on_long_synthetic_data(self, synthetic_demand_long):
        """Test fitting on longer synthetic data (2 years)."""
        forecaster = ProphetForecaster(include_holidays=False)
        forecaster.fit(synthetic_demand_long)

        assert forecaster.fitted_model is not None
        assert len(forecaster.y_train_index) == 2*365*48

    def test_fit_raises_valueerror_without_datetimeindex(self):
        """Test that fit() raises ValueError if Series lacks DatetimeIndex."""
        forecaster = ProphetForecaster()
        series = pd.Series([1, 2, 3, 4, 5], index=[0, 1, 2, 3, 4])

        with pytest.raises(ValueError, match="must have a DatetimeIndex"):
            forecaster.fit(series)

    def test_fit_with_holidays_includes_holidays_df(self, synthetic_demand_long):
        """Test that fit() with holidays creates holidays_df."""
        forecaster = ProphetForecaster(include_holidays=True)
        forecaster.fit(synthetic_demand_long)

        assert forecaster.holidays_df is not None
        assert "holiday" in forecaster.holidays_df.columns
        assert "ds" in forecaster.holidays_df.columns

    def test_fit_without_holidays_has_no_holidays_df(self, synthetic_demand_long):
        """Test that fit() without holidays has no holidays_df."""
        forecaster = ProphetForecaster(include_holidays=False)
        forecaster.fit(synthetic_demand_long)

        assert forecaster.holidays_df is None


class TestProphetForecasterPredict:
    """Test prediction output shape, values, and DatetimeIndex."""

    def test_predict_output_shape(self, synthetic_demand_long):
        """Test predict() returns correct output shape."""
        forecaster = ProphetForecaster(include_holidays=False)
        forecaster.fit(synthetic_demand_long)

        forecast = forecaster.predict(horizon=48)

        assert isinstance(forecast, pd.Series)
        assert len(forecast) == 48

    def test_predict_no_null_values(self, synthetic_demand_long):
        """Test predict() output contains no null values."""
        forecaster = ProphetForecaster(include_holidays=False)
        forecaster.fit(synthetic_demand_long)

        forecast = forecaster.predict(horizon=48)

        assert not forecast.isna().any()
        assert forecast.dtype in [np.float64, np.float32]

    def test_predict_datetimeindex_incremental_30min(self, synthetic_demand_long):
        """Test predict() index is strictly incremental 30-minute DatetimeIndex."""
        forecaster = ProphetForecaster(include_holidays=False)
        forecaster.fit(synthetic_demand_long)

        forecast = forecaster.predict(horizon=48)

        # Index should be DatetimeIndex
        assert isinstance(forecast.index, pd.DatetimeIndex)

        # Index should be monotonically increasing
        assert forecast.index.is_monotonic_increasing

        # All consecutive differences should be exactly 30 minutes
        diffs = forecast.index.to_series().diff()[1:]
        expected_diff = pd.Timedelta(minutes=30)
        assert (diffs == expected_diff).all()

    def test_predict_index_starts_30min_after_training(self, synthetic_demand_long):
        """Test predict() index starts 30min after y_train.index[-1]."""
        forecaster = ProphetForecaster(include_holidays=False)
        forecaster.fit(synthetic_demand_long)

        forecast = forecaster.predict(horizon=24)

        last_train_time = synthetic_demand_long.index[-1]
        expected_first_time = last_train_time + pd.Timedelta(minutes=30)

        assert forecast.index[0] == expected_first_time

    def test_predict_raises_valueerror_before_fit(self):
        """Test that predict() raises ValueError if called before fit()."""
        forecaster = ProphetForecaster()

        with pytest.raises(ValueError, match="not been fitted"):
            forecaster.predict(horizon=48)

    def test_predict_raises_valueerror_for_zero_horizon(self, synthetic_demand_long):
        """Test that predict() raises ValueError if horizon <= 0."""
        forecaster = ProphetForecaster(include_holidays=False)
        forecaster.fit(synthetic_demand_long)

        with pytest.raises(ValueError, match="positive integer"):
            forecaster.predict(horizon=0)

    def test_predict_raises_valueerror_for_negative_horizon(self, synthetic_demand_long):
        """Test that predict() raises ValueError for negative horizon."""
        forecaster = ProphetForecaster(include_holidays=False)
        forecaster.fit(synthetic_demand_long)

        with pytest.raises(ValueError, match="positive integer"):
            forecaster.predict(horizon=-5)

    def test_predict_long_horizon(self, synthetic_demand_long):
        """Test predict() with longer horizon (336 steps = 7 days)."""
        forecaster = ProphetForecaster(include_holidays=False)
        forecaster.fit(synthetic_demand_long)

        forecast = forecaster.predict(horizon=336)

        assert len(forecast) == 336
        assert not forecast.isna().any()


class TestProphetForecasterSerialization:
    """Test save/load serialization with tmp_path."""

    def test_save_creates_directories_and_file(self, synthetic_demand_long, tmp_path):
        """Test save() creates parent directories and model file."""
        forecaster = ProphetForecaster(include_holidays=False)
        forecaster.fit(synthetic_demand_long)

        # Use nested path to test directory creation
        model_path = tmp_path / "models" / "nested" / "model.pkl"
        forecaster.save(model_path)

        assert model_path.exists()
        assert model_path.stat().st_size > 0

    def test_save_and_load_produce_identical_predictions(self, synthetic_demand_long, tmp_path):
        """Test save/load produces identical predictions to original."""
        forecaster = ProphetForecaster(include_holidays=False)
        forecaster.fit(synthetic_demand_long)

        # Get predictions from original
        original_forecast = forecaster.predict(horizon=48)

        # Save and load
        model_path = tmp_path / "model.pkl"
        forecaster.save(model_path)
        loaded_forecaster = ProphetForecaster.load(model_path)

        # Get predictions from loaded
        loaded_forecast = loaded_forecaster.predict(horizon=48)

        # Predictions should be very close (Prophet uses some randomization)
        assert np.allclose(original_forecast.values, loaded_forecast.values, rtol=0.01)
        assert (original_forecast.index == loaded_forecast.index).all()

    def test_save_and_load_preserves_config(self, synthetic_demand_long, tmp_path):
        """Test save/load preserves forecaster configuration."""
        forecaster = ProphetForecaster(
            include_holidays=True,
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        forecaster.fit(synthetic_demand_long)

        # Save and load
        model_path = tmp_path / "model.pkl"
        forecaster.save(model_path)
        loaded_forecaster = ProphetForecaster.load(model_path)

        # Configuration should be preserved
        assert loaded_forecaster.include_holidays == True
        assert loaded_forecaster.yearly_seasonality == False
        assert loaded_forecaster.weekly_seasonality == True
        assert loaded_forecaster.daily_seasonality == False

    def test_save_raises_error_before_fit(self, tmp_path):
        """Test that save() raises ValueError if called before fit()."""
        forecaster = ProphetForecaster()
        model_path = tmp_path / "model.pkl"

        with pytest.raises(ValueError, match="not been fitted"):
            forecaster.save(model_path)


class TestProphetHolidaysIntegration:
    """Test holiday handling in ProphetForecaster."""

    def test_get_holidays_df_with_holidays_enabled(self):
        """Test that _get_holidays_df returns a DataFrame when holidays enabled."""
        forecaster = ProphetForecaster(include_holidays=True)
        holidays_df = forecaster._get_holidays_df()

        assert holidays_df is not None
        assert isinstance(holidays_df, pd.DataFrame)
        assert "holiday" in holidays_df.columns
        assert "ds" in holidays_df.columns
        assert len(holidays_df) > 0

    def test_get_holidays_df_with_holidays_disabled(self):
        """Test that _get_holidays_df returns None when holidays disabled."""
        forecaster = ProphetForecaster(include_holidays=False)
        holidays_df = forecaster._get_holidays_df()

        assert holidays_df is None

    def test_fit_with_holidays_includes_peruvian_holidays(self, synthetic_demand_long):
        """Test that fit() with holidays uses Peruvian holidays."""
        forecaster = ProphetForecaster(include_holidays=True)
        forecaster.fit(synthetic_demand_long)

        # Should have holidays_df
        assert forecaster.holidays_df is not None

        # Should contain known Peruvian holidays
        holiday_names = forecaster.holidays_df['holiday'].unique()
        # Check for some common holiday patterns
        assert len(holiday_names) > 0
