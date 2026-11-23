from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.base import BaseForecaster

# One week of 30-minute steps: the seasonal period used by the naive forecast.
SEASON_STEPS = 336


class SeasonalNaiveForecaster(BaseForecaster):
    """
    Seasonal naive forecaster with a weekly season.

    The last SEASON_STEPS (336) training values are stored and tiled across the
    forecast horizon, so each forecast value equals the value 336 steps earlier.
    """

    def __init__(self):
        self.last_values: np.ndarray | None = None
        self.last_timestamp: pd.Timestamp | None = None

    def fit(self, y_train: pd.Series, X_train: pd.DataFrame | None = None) -> "SeasonalNaiveForecaster":
        """
        Store the last week of training values and the last training timestamp.

        Parameters
        ----------
        y_train : pd.Series
            Historical target values with DatetimeIndex (30-minute steps).
        X_train : pd.DataFrame | None
            Ignored; accepted for interface compatibility.

        Returns
        -------
        SeasonalNaiveForecaster
            Self for method chaining.

        Raises
        ------
        TypeError
            If y_train lacks a DatetimeIndex.
        ValueError
            If y_train has fewer than 336 values.
        """
        if not isinstance(y_train.index, pd.DatetimeIndex):
            raise TypeError("y_train must have a DatetimeIndex")
        if len(y_train) < SEASON_STEPS:
            raise ValueError(
                f"y_train needs at least {SEASON_STEPS} values, got {len(y_train)}"
            )

        self.last_values = y_train.to_numpy()[-SEASON_STEPS:].copy()
        self.last_timestamp = y_train.index[-1]
        return self

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        """
        Forecast by repeating the last week of training values.

        Parameters
        ----------
        horizon : int
            Number of 30-minute steps to forecast. Must be positive.
        X_future : pd.DataFrame | None
            Ignored; accepted for interface compatibility.

        Returns
        -------
        pd.Series
            Forecast values indexed by the future 30-minute DatetimeIndex.

        Raises
        ------
        ValueError
            If the model has not been fitted or horizon <= 0.
        """
        if self.last_values is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        if horizon <= 0:
            raise ValueError("horizon must be a positive integer")

        values = self.last_values[np.arange(horizon) % SEASON_STEPS]
        future_index = pd.date_range(
            start=self.last_timestamp + pd.Timedelta(minutes=30),
            periods=horizon,
            freq="30min",
        )
        return pd.Series(values, index=future_index, name="Demand")

    def save(self, path: str | Path) -> None:
        """
        Save the fitted model to disk using joblib.

        Parameters
        ----------
        path : str | Path
            File path where the model should be saved.

        Raises
        ------
        ValueError
            If the model has not been fitted.
        """
        if self.last_values is None:
            raise ValueError("Model has not been fitted. Nothing to save.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "last_values": self.last_values,
            "last_timestamp": self.last_timestamp,
        }
        joblib.dump(model_data, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "SeasonalNaiveForecaster":
        """
        Load a fitted model from disk.

        Parameters
        ----------
        path : str | Path
            File path to the saved model.

        Returns
        -------
        SeasonalNaiveForecaster
            Deserialized model object ready for prediction.

        Raises
        ------
        FileNotFoundError
            If the model file does not exist.
        """
        try:
            model_data = joblib.load(str(path))
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {path}")

        forecaster = cls()
        forecaster.last_values = model_data["last_values"]
        forecaster.last_timestamp = model_data["last_timestamp"]
        return forecaster
