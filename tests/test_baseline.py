import numpy as np
import pandas as pd
import pytest

from src.models.baseline import SeasonalNaiveForecaster


@pytest.fixture
def demand():
    """1000 steps (~3.5 weeks) of 30-minute demand with a weekly pattern and noise."""
    idx = pd.date_range("2025-01-01", periods=1000, freq="30min")
    rng = np.random.default_rng(42)
    hours = idx.hour + idx.minute / 60 + idx.dayofweek * 24
    values = 100 + 30 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 2, len(idx))
    return pd.Series(values, index=idx, name="Demand")


class TestSeasonalNaiveOutput:
    def test_length_and_index(self, demand):
        forecast = SeasonalNaiveForecaster().fit(demand).predict(1440)

        assert len(forecast) == 1440
        expected_index = pd.date_range(
            demand.index[-1] + pd.Timedelta(minutes=30), periods=1440, freq="30min"
        )
        pd.testing.assert_index_equal(forecast.index, expected_index)

    def test_values_equal_values_336_steps_earlier(self, demand):
        n = len(demand)
        forecast = SeasonalNaiveForecaster().fit(demand).predict(1440)

        # First week of forecast repeats the last week of training data.
        np.testing.assert_array_equal(forecast.to_numpy()[:336], demand.to_numpy()[-336:])

        # Every forecast step equals the value 336 steps earlier in the joined series.
        full = np.concatenate([demand.to_numpy(), forecast.to_numpy()])
        np.testing.assert_array_equal(full[n:], full[n - 336 : n + 1440 - 336])


class TestSeasonalNaivePersistence:
    def test_save_load_gives_identical_predictions(self, demand, tmp_path):
        model = SeasonalNaiveForecaster().fit(demand)
        path = tmp_path / "seasonal_naive.joblib"
        model.save(path)

        loaded = SeasonalNaiveForecaster.load(path)

        pd.testing.assert_series_equal(model.predict(500), loaded.predict(500))

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SeasonalNaiveForecaster.load(tmp_path / "missing.joblib")

    def test_save_before_fit_raises(self, tmp_path):
        with pytest.raises(ValueError, match="fit"):
            SeasonalNaiveForecaster().save(tmp_path / "model.joblib")


class TestSeasonalNaiveErrors:
    def test_predict_before_fit_raises(self):
        with pytest.raises(ValueError, match="fit"):
            SeasonalNaiveForecaster().predict(10)

    def test_too_little_data_raises(self, demand):
        with pytest.raises(ValueError, match="336"):
            SeasonalNaiveForecaster().fit(demand.iloc[:335])

    def test_exactly_336_values_is_enough(self, demand):
        forecast = SeasonalNaiveForecaster().fit(demand.iloc[:336]).predict(48)
        assert len(forecast) == 48

    def test_requires_datetime_index(self, demand):
        plain = pd.Series(demand.to_numpy())
        with pytest.raises(TypeError, match="DatetimeIndex"):
            SeasonalNaiveForecaster().fit(plain)

    @pytest.mark.parametrize("horizon", [0, -5])
    def test_non_positive_horizon_raises(self, demand, horizon):
        with pytest.raises(ValueError, match="horizon"):
            SeasonalNaiveForecaster().fit(demand).predict(horizon)
