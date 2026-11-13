import numpy as np
import pandas as pd
import pytest
from src.models.lstm_model import LSTMForecaster

# Tiny configuration so the tests stay fast on CPU.
TINY = dict(
    lookback=24,
    hidden_dim=8,
    num_layers=1,
    dropout=0.0,
    epochs=2,
    batch_size=16,
    lr=1e-2,
    block_size=8,
    seed=0,
)


def make_synthetic_demand(periods: int = 300) -> pd.Series:
    """Synthetic 30-minute demand with a daily cycle and small noise."""
    dates = pd.date_range("2024-01-01", periods=periods, freq="30min")
    hours = dates.hour + dates.minute / 60
    rng = np.random.default_rng(0)
    values = 100 + 30 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 2, len(dates))
    return pd.Series(values, index=dates, name="Demand")


@pytest.fixture
def y_train():
    return make_synthetic_demand()


@pytest.fixture(scope="module")
def fitted_forecaster():
    return LSTMForecaster(**TINY).fit(make_synthetic_demand())


class TestLSTMInit:
    """Constructor validation."""

    def test_invalid_num_layers_raises(self):
        with pytest.raises(ValueError, match="num_layers must be 1 or 2"):
            LSTMForecaster(num_layers=3)

    def test_default_hyperparameters(self):
        forecaster = LSTMForecaster()
        assert forecaster.lookback == 336
        assert forecaster.hidden_dim == 64
        assert forecaster.block_size == 48
        assert forecaster.net is None


class TestLSTMFit:
    """Fitting behaviour and input checks."""

    def test_fit_returns_self(self, y_train):
        forecaster = LSTMForecaster(**TINY)
        assert forecaster.fit(y_train) is forecaster
        assert forecaster.net is not None
        assert forecaster.epochs_run_ == TINY["epochs"]  # patience exceeds 2 epochs, so no early stop

    def test_fit_raises_without_datetimeindex(self):
        series = pd.Series(np.arange(300, dtype=float))
        with pytest.raises(ValueError, match="must have a DatetimeIndex"):
            LSTMForecaster(**TINY).fit(series)

    def test_fit_raises_when_series_not_longer_than_lookback(self):
        short = make_synthetic_demand(periods=TINY["lookback"])
        with pytest.raises(ValueError, match="more than lookback"):
            LSTMForecaster(**TINY).fit(short)

    def test_fit_raises_when_too_short_for_one_window(self):
        # Long enough for lookback, too short for lookback + block_size after the 10% split.
        short = make_synthetic_demand(periods=TINY["lookback"] + 2)
        with pytest.raises(ValueError, match="too short for one training window"):
            LSTMForecaster(**TINY).fit(short)

    def test_scaler_fit_on_train_only(self, y_train):
        forecaster = LSTMForecaster(**TINY).fit(y_train)
        assert forecaster.scaler.mean_[0] == pytest.approx(y_train.mean())
        assert forecaster.scaler.scale_[0] == pytest.approx(y_train.std(ddof=0))


class TestLSTMPredict:
    """Prediction shape, index and error handling."""

    @pytest.mark.parametrize("horizon", [1, 8, 20])
    def test_predict_output_shape(self, fitted_forecaster, horizon):
        # 20 is not a multiple of block_size=8, so the last block must be trimmed.
        forecast = fitted_forecaster.predict(horizon)
        assert isinstance(forecast, pd.Series)
        assert len(forecast) == horizon
        assert np.all(np.isfinite(forecast.to_numpy()))

    def test_predict_returns_future_30min_datetimeindex(self, fitted_forecaster):
        forecast = fitted_forecaster.predict(20)
        index = forecast.index
        assert isinstance(index, pd.DatetimeIndex)
        assert index[0] == fitted_forecaster.y_train_index[-1] + pd.Timedelta(minutes=30)
        assert index.is_monotonic_increasing
        assert np.all(np.diff(index.asi8) == pd.Timedelta(minutes=30).value)

    def test_predict_raises_before_fit(self):
        with pytest.raises(ValueError, match="Call fit"):
            LSTMForecaster(**TINY).predict(10)

    @pytest.mark.parametrize("horizon", [0, -5])
    def test_predict_rejects_non_positive_horizon(self, fitted_forecaster, horizon):
        with pytest.raises(ValueError, match="positive integer"):
            fitted_forecaster.predict(horizon)


class TestLSTMSaveLoad:
    """Serialization round-trip."""

    def test_save_raises_before_fit(self, tmp_path):
        with pytest.raises(ValueError, match="Nothing to save"):
            LSTMForecaster(**TINY).save(tmp_path / "lstm.pt")

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            LSTMForecaster.load(tmp_path / "missing.pt")

    def test_save_load_identical_predictions(self, fitted_forecaster, tmp_path):
        path = tmp_path / "nested" / "lstm.pt"
        fitted_forecaster.save(path)
        loaded = LSTMForecaster.load(path)

        np.testing.assert_allclose(
            loaded.predict(20).to_numpy(),
            fitted_forecaster.predict(20).to_numpy(),
            rtol=1e-6,
        )

    def test_load_restores_hyperparameters_and_context(self, fitted_forecaster, tmp_path):
        path = tmp_path / "lstm.pt"
        fitted_forecaster.save(path)
        loaded = LSTMForecaster.load(path)

        assert loaded._hyperparameters() == fitted_forecaster._hyperparameters()
        pd.testing.assert_index_equal(loaded.y_train_index, fitted_forecaster.y_train_index)
        np.testing.assert_array_equal(loaded.last_values, fitted_forecaster.last_values)
        assert loaded.scaler.mean_[0] == pytest.approx(fitted_forecaster.scaler.mean_[0])


class TestLSTMReproducibility:
    """Fixed seeds."""

    def test_same_seed_same_predictions(self, y_train):
        first = LSTMForecaster(**TINY).fit(y_train).predict(16)
        second = LSTMForecaster(**TINY).fit(y_train).predict(16)
        np.testing.assert_allclose(first.to_numpy(), second.to_numpy(), rtol=1e-6)
