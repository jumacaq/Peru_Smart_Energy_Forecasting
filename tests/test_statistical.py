import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.models.statistical import HoltWintersForecaster


@pytest.fixture
def synthetic_demand_96steps():
    """Create synthetic 30-minute demand data covering 96 steps (2 full days)."""
    dates = pd.date_range("2024-01-01", periods=96, freq="30min")  # 2 days
    # Create synthetic demand with daily pattern (2 complete cycles)
    hours = dates.hour + dates.minute / 60
    demand = 100 + 30 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 2, len(dates))
    series = pd.Series(demand, index=dates, name="Demand")
    return series


@pytest.fixture
def synthetic_demand_long():
    """Create longer synthetic demand data for extended training."""
    dates = pd.date_range("2024-01-01", periods=336, freq="30min")  # 7 days
    hours = dates.hour + dates.minute / 60
    demand = 100 + 30 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 3, len(dates))
    series = pd.Series(demand, index=dates, name="Demand")
    return series


class TestHoltWintersForecasterInit:
    """Test model initialization."""

    def test_initialization_default_params(self):
        """Test initialization with default parameters."""
        forecaster = HoltWintersForecaster()
        assert forecaster.seasonal_periods == 48
        assert forecaster.trend == "add"
        assert forecaster.seasonal == "add"
        assert forecaster.damped_trend is True
        assert forecaster.fitted_model is None

    def test_initialization_with_custom_damped_trend(self):
        """Test initialization with custom damped_trend parameter."""
        forecaster = HoltWintersForecaster(damped_trend=False)
        assert forecaster.damped_trend is False

        forecaster_damped = HoltWintersForecaster(damped_trend=True)
        assert forecaster_damped.damped_trend is True


class TestHoltWintersForecasterFit:
    """Test fitting on synthetic demand data."""

    def test_fit_on_96step_synthetic_data(self, synthetic_demand_96steps):
        """Test fitting on 2 full days (96 steps) of synthetic data."""
        forecaster = HoltWintersForecaster()
        result = forecaster.fit(synthetic_demand_96steps)

        # Should return self for method chaining
        assert result is forecaster

        # Should store fitted model
        assert forecaster.fitted_model is not None

        # Should store training index
        assert forecaster.y_train_index is not None
        assert len(forecaster.y_train_index) == 96

    def test_fit_raises_valueerror_without_datetimeindex(self):
        """Test that fit() raises ValueError if Series lacks DatetimeIndex."""
        forecaster = HoltWintersForecaster()
        series = pd.Series([1, 2, 3, 4, 5], index=[0, 1, 2, 3, 4])

        with pytest.raises(ValueError, match="must have a DatetimeIndex"):
            forecaster.fit(series)


class TestHoltWintersForecasterPredict:
    """Test prediction output shape, values, and DatetimeIndex."""

    def test_predict_output_shape(self, synthetic_demand_96steps):
        """Test predict() returns correct output shape."""
        forecaster = HoltWintersForecaster()
        forecaster.fit(synthetic_demand_96steps)

        forecast = forecaster.predict(horizon=48)

        assert isinstance(forecast, pd.Series)
        assert len(forecast) == 48

    def test_predict_no_null_values(self, synthetic_demand_96steps):
        """Test predict() output contains no null values."""
        forecaster = HoltWintersForecaster()
        forecaster.fit(synthetic_demand_96steps)

        forecast = forecaster.predict(horizon=48)

        assert not forecast.isna().any()
        assert forecast.dtype in [np.float64, np.float32]

    def test_predict_datetimeindex_incremental_30min(self, synthetic_demand_96steps):
        """Test predict() index is strictly incremental 30-minute DatetimeIndex."""
        forecaster = HoltWintersForecaster()
        forecaster.fit(synthetic_demand_96steps)

        forecast = forecaster.predict(horizon=48)

        # Index should be DatetimeIndex
        assert isinstance(forecast.index, pd.DatetimeIndex)

        # Index should be monotonically increasing
        assert forecast.index.is_monotonic_increasing

        # All consecutive differences should be exactly 30 minutes
        diffs = forecast.index.to_series().diff()[1:]
        expected_diff = pd.Timedelta(minutes=30)
        assert (diffs == expected_diff).all()

    def test_predict_index_starts_30min_after_training(self, synthetic_demand_96steps):
        """Test predict() index starts 30min after y_train.index[-1]."""
        forecaster = HoltWintersForecaster()
        forecaster.fit(synthetic_demand_96steps)

        forecast = forecaster.predict(horizon=24)

        last_train_time = synthetic_demand_96steps.index[-1]
        expected_first_time = last_train_time + pd.Timedelta(minutes=30)

        assert forecast.index[0] == expected_first_time

    def test_predict_raises_valueerror_before_fit(self):
        """Test that predict() raises ValueError if called before fit()."""
        forecaster = HoltWintersForecaster()

        with pytest.raises(ValueError, match="not been fitted"):
            forecaster.predict(horizon=48)

    def test_predict_raises_valueerror_for_zero_horizon(self, synthetic_demand_96steps):
        """Test that predict() raises ValueError if horizon <= 0."""
        forecaster = HoltWintersForecaster()
        forecaster.fit(synthetic_demand_96steps)

        with pytest.raises(ValueError, match="positive integer"):
            forecaster.predict(horizon=0)

    def test_predict_raises_valueerror_for_negative_horizon(self, synthetic_demand_96steps):
        """Test that predict() raises ValueError for negative horizon."""
        forecaster = HoltWintersForecaster()
        forecaster.fit(synthetic_demand_96steps)

        with pytest.raises(ValueError, match="positive integer"):
            forecaster.predict(horizon=-5)


class TestHoltWintersForecasterSerialization:
    """Test save/load serialization with tmp_path."""

    def test_save_creates_directories_and_file(self, synthetic_demand_96steps, tmp_path):
        """Test save() creates parent directories and model file."""
        forecaster = HoltWintersForecaster()
        forecaster.fit(synthetic_demand_96steps)

        # Use nested path to test directory creation
        model_path = tmp_path / "models" / "nested" / "model.pkl"
        forecaster.save(model_path)

        assert model_path.exists()
        assert model_path.stat().st_size > 0

    def test_save_and_load_produce_identical_predictions(self, synthetic_demand_96steps, tmp_path):
        """Test save/load produces identical predictions to original."""
        forecaster = HoltWintersForecaster()
        forecaster.fit(synthetic_demand_96steps)

        # Get predictions from original
        original_forecast = forecaster.predict(horizon=48)

        # Save and load
        model_path = tmp_path / "model.pkl"
        forecaster.save(model_path)
        loaded_forecaster = HoltWintersForecaster.load(model_path)

        # Get predictions from loaded
        loaded_forecast = loaded_forecaster.predict(horizon=48)

        # Predictions should be identical
        assert np.allclose(original_forecast.values, loaded_forecast.values, rtol=1e-10)
        assert (original_forecast.index == loaded_forecast.index).all()

    def test_save_and_load_preserves_damped_trend(self, synthetic_demand_96steps, tmp_path):
        """Test save/load preserves damped_trend parameter."""
        forecaster = HoltWintersForecaster(damped_trend=False)
        forecaster.fit(synthetic_demand_96steps)

        # Save and load
        model_path = tmp_path / "model.pkl"
        forecaster.save(model_path)
        loaded_forecaster = HoltWintersForecaster.load(model_path)

        # damped_trend should be preserved
        assert loaded_forecaster.damped_trend is False

    def test_save_and_load_handles_legacy_models(self, synthetic_demand_96steps, tmp_path):
        """Test load() handles models saved without damped_trend (backward compat)."""
        forecaster = HoltWintersForecaster(damped_trend=True)
        forecaster.fit(synthetic_demand_96steps)

        # Save model and manually remove damped_trend from saved data to simulate legacy model
        model_path = tmp_path / "legacy_model.pkl"
        forecaster.save(model_path)

        # Load and remove damped_trend field to simulate legacy saved model
        import joblib
        model_data = joblib.load(str(model_path))
        del model_data["damped_trend"]
        joblib.dump(model_data, str(model_path))

        # Loading should work with default damped_trend=True
        loaded_forecaster = HoltWintersForecaster.load(model_path)
        assert loaded_forecaster.damped_trend is True
