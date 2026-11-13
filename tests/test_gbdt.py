import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.models.gbdt import LightGBMForecaster


@pytest.fixture
def synthetic_demand_long():
    """Create longer synthetic demand data for LightGBM training (2+ years)."""
    dates = pd.date_range("2022-01-01", periods=2*365*48, freq="30min")
    hours = dates.hour + dates.minute / 60
    demand = 100 + 30 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 3, len(dates))
    series = pd.Series(demand, index=dates, name="Demand")
    return series


class TestLightGBMForecasterInit:
    """Test model initialization."""

    def test_initialization_default_params(self):
        """Test initialization with default parameters."""
        forecaster = LightGBMForecaster()
        assert forecaster.num_leaves == 31
        assert forecaster.learning_rate == 0.05
        assert forecaster.n_estimators == 100
        assert forecaster.forecaster is None

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        forecaster = LightGBMForecaster(
            num_leaves=15,
            learning_rate=0.1,
            n_estimators=200
        )
        assert forecaster.num_leaves == 15
        assert forecaster.learning_rate == 0.1
        assert forecaster.n_estimators == 200


class TestLightGBMForecasterFit:
    """Test fitting on synthetic demand data."""

    def test_fit_on_long_synthetic_data(self, synthetic_demand_long):
        """Test fitting on longer synthetic data."""
        forecaster = LightGBMForecaster(n_estimators=50)
        result = forecaster.fit(synthetic_demand_long)

        assert result is forecaster
        assert forecaster.forecaster is not None
        assert forecaster.y_train_index is not None

    def test_fit_raises_valueerror_without_datetimeindex(self):
        """Test that fit() raises ValueError if Series lacks DatetimeIndex."""
        forecaster = LightGBMForecaster()
        series = pd.Series([1, 2, 3, 4, 5], index=[0, 1, 2, 3, 4])

        with pytest.raises(ValueError, match="must have a DatetimeIndex"):
            forecaster.fit(series)


class TestLightGBMForecasterPredict:
    """Test prediction output shape, values, and DatetimeIndex."""

    def test_predict_output_shape(self, synthetic_demand_long):
        """Test predict() returns correct output shape."""
        forecaster = LightGBMForecaster(n_estimators=30)
        forecaster.fit(synthetic_demand_long)

        forecast = forecaster.predict(horizon=48)

        assert isinstance(forecast, pd.Series)
        assert len(forecast) == 48

    def test_predict_no_null_values(self, synthetic_demand_long):
        """Test predict() output contains no null values."""
        forecaster = LightGBMForecaster(n_estimators=30)
        forecaster.fit(synthetic_demand_long)

        forecast = forecaster.predict(horizon=48)

        assert not forecast.isna().any()
        assert forecast.dtype in [np.float64, np.float32]

    def test_predict_datetimeindex_incremental_30min(self, synthetic_demand_long):
        """Test predict() index is strictly incremental 30-minute DatetimeIndex."""
        forecaster = LightGBMForecaster(n_estimators=30)
        forecaster.fit(synthetic_demand_long)

        forecast = forecaster.predict(horizon=48)

        assert isinstance(forecast.index, pd.DatetimeIndex)
        assert forecast.index.is_monotonic_increasing

        diffs = forecast.index.to_series().diff()[1:]
        expected_diff = pd.Timedelta(minutes=30)
        assert (diffs == expected_diff).all()

    def test_predict_index_starts_30min_after_training(self, synthetic_demand_long):
        """Test predict() index starts 30min after y_train.index[-1]."""
        forecaster = LightGBMForecaster(n_estimators=30)
        forecaster.fit(synthetic_demand_long)

        forecast = forecaster.predict(horizon=24)

        last_train_time = synthetic_demand_long.index[-1]
        expected_first_time = last_train_time + pd.Timedelta(minutes=30)

        assert forecast.index[0] == expected_first_time

    def test_predict_raises_valueerror_before_fit(self):
        """Test that predict() raises ValueError if called before fit()."""
        forecaster = LightGBMForecaster()

        with pytest.raises(ValueError, match="not been fitted"):
            forecaster.predict(horizon=48)

    def test_predict_raises_valueerror_for_zero_horizon(self, synthetic_demand_long):
        """Test that predict() raises ValueError if horizon <= 0."""
        forecaster = LightGBMForecaster(n_estimators=30)
        forecaster.fit(synthetic_demand_long)

        with pytest.raises(ValueError, match="positive integer"):
            forecaster.predict(horizon=0)

    def test_predict_raises_valueerror_for_negative_horizon(self, synthetic_demand_long):
        """Test that predict() raises ValueError for negative horizon."""
        forecaster = LightGBMForecaster(n_estimators=30)
        forecaster.fit(synthetic_demand_long)

        with pytest.raises(ValueError, match="positive integer"):
            forecaster.predict(horizon=-5)

    def test_predict_long_horizon(self, synthetic_demand_long):
        """Test predict() with longer horizon (336 steps = 7 days)."""
        forecaster = LightGBMForecaster(n_estimators=30)
        forecaster.fit(synthetic_demand_long)

        forecast = forecaster.predict(horizon=336)

        assert len(forecast) == 336
        assert not forecast.isna().any()


class TestLightGBMForecasterExogenousFeatures:
    """Test exogenous feature consistency."""

    def test_exog_columns_match_between_train_and_predict(self, synthetic_demand_long):
        """Test that exogenous feature columns match between fit and predict."""
        forecaster = LightGBMForecaster(n_estimators=30)
        forecaster.fit(synthetic_demand_long)

        # Generate exog for training period
        exog_train = forecaster._get_calendar_features(synthetic_demand_long.index)

        # Generate exog for a future horizon
        last_timestamp = synthetic_demand_long.index[-1]
        future_index = pd.date_range(
            start=last_timestamp + pd.Timedelta(minutes=30),
            periods=48,
            freq="30min"
        )
        exog_future = forecaster._get_calendar_features(future_index)

        # Columns should match exactly
        assert exog_train.columns.tolist() == exog_future.columns.tolist(), \
            f"Exog columns mismatch: train {exog_train.columns.tolist()} vs future {exog_future.columns.tolist()}"

        # All columns should be non-empty
        assert len(exog_train.columns) > 0, "No exog features generated"
        assert len(exog_future.columns) > 0, "No exog features generated"


class TestLightGBMForecasterSerialization:
    """Test save/load serialization with tmp_path."""

    def test_save_creates_directories_and_file(self, synthetic_demand_long, tmp_path):
        """Test save() creates parent directories and model file."""
        forecaster = LightGBMForecaster(n_estimators=30)
        forecaster.fit(synthetic_demand_long)

        model_path = tmp_path / "models" / "nested" / "model.pkl"
        forecaster.save(model_path)

        assert model_path.exists()
        assert model_path.stat().st_size > 0

    def test_save_and_load_produce_identical_predictions(self, synthetic_demand_long, tmp_path):
        """Test save/load produces identical predictions to original."""
        forecaster = LightGBMForecaster(n_estimators=30)
        forecaster.fit(synthetic_demand_long)

        # Get predictions from original
        original_forecast = forecaster.predict(horizon=48)

        # Save and load
        model_path = tmp_path / "model.pkl"
        forecaster.save(model_path)
        loaded_forecaster = LightGBMForecaster.load(model_path)

        # Get predictions from loaded
        loaded_forecast = loaded_forecaster.predict(horizon=48)

        # Predictions should be identical
        assert np.allclose(original_forecast.values, loaded_forecast.values, rtol=1e-5)
        assert (original_forecast.index == loaded_forecast.index).all()

    def test_save_and_load_preserves_config(self, synthetic_demand_long, tmp_path):
        """Test save/load preserves forecaster configuration."""
        forecaster = LightGBMForecaster(
            num_leaves=15,
            learning_rate=0.1,
            n_estimators=50
        )
        forecaster.fit(synthetic_demand_long)

        model_path = tmp_path / "model.pkl"
        forecaster.save(model_path)
        loaded_forecaster = LightGBMForecaster.load(model_path)

        assert loaded_forecaster.num_leaves == 15
        assert loaded_forecaster.learning_rate == 0.1
        assert loaded_forecaster.n_estimators == 50

    def test_save_raises_error_before_fit(self, tmp_path):
        """Test that save() raises ValueError if called before fit()."""
        forecaster = LightGBMForecaster()
        model_path = tmp_path / "model.pkl"

        with pytest.raises(ValueError, match="not been fitted"):
            forecaster.save(model_path)
