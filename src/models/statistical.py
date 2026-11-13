from pathlib import Path
import pandas as pd
import joblib
from statsmodels.tsa.api import ExponentialSmoothing
from src.models.base import BaseForecaster


class HoltWintersForecaster(BaseForecaster):
    """
    Holt-Winters Exponential Smoothing forecaster for time series prediction.

    Handles single seasonality (seasonal_periods=48 for 30-minute intervals covering daily patterns).
    Supports damped trend for improved long-term forecasting stability.
    """

    def __init__(self, seasonal_periods: int = 48, trend: str = "add", seasonal: str = "add", damped_trend: bool = True):
        """
        Initialize the Holt-Winters forecaster.

        Parameters
        ----------
        seasonal_periods : int
            Number of periods in a season. Default 48 for 30-minute intervals (24 hours).
        trend : str
            Trend component: "add" for additive, "mul" for multiplicative.
        seasonal : str
            Seasonal component: "add" for additive, "mul" for multiplicative.
        damped_trend : bool
            Whether to use damped trend. Default True for more stable long-term forecasts.
        """
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self.damped_trend = damped_trend
        self.model = None
        self.fitted_model = None
        self.y_train_index = None

    def fit(self, y_train: pd.Series, X_train: pd.DataFrame | None = None) -> "HoltWintersForecaster":
        """
        Fit the Holt-Winters Exponential Smoothing model.

        Parameters
        ----------
        y_train : pd.Series
            Historical target values with DatetimeIndex.
        X_train : pd.DataFrame | None
            Optional exogenous features (ignored for statistical models).

        Returns
        -------
        HoltWintersForecaster
            Self for method chaining.

        Raises
        ------
        ValueError
            If y_train lacks a DatetimeIndex.
        """
        if not isinstance(y_train.index, pd.DatetimeIndex):
            raise ValueError("y_train must have a DatetimeIndex")

        # Store the training index for prediction
        self.y_train_index = y_train.index

        # Fit the ExponentialSmoothing model
        self.model = ExponentialSmoothing(
            y_train,
            seasonal_periods=self.seasonal_periods,
            trend=self.trend,
            seasonal=self.seasonal,
            damped_trend=self.damped_trend,
            initialization_method="estimated"
        )
        self.fitted_model = self.model.fit(optimized=True)

        return self

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        """
        Forecast future values.

        Parameters
        ----------
        horizon : int
            Number of 30-minute steps to forecast.
        X_future : pd.DataFrame | None
            Optional exogenous features (ignored for statistical models).

        Returns
        -------
        pd.Series
            Forecasted values with valid future 30-minute DatetimeIndex.

        Raises
        ------
        ValueError
            If the model has not been fitted or if horizon <= 0.
        RuntimeError
            If y_train lacks a DatetimeIndex.
        """
        if self.fitted_model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        if horizon <= 0:
            raise ValueError("horizon must be a positive integer")

        if self.y_train_index is None:
            raise RuntimeError("Training data index not available")

        # Get forecast
        forecast_values = self.fitted_model.forecast(steps=horizon)

        # Build future DatetimeIndex starting after the last training index
        last_timestamp = self.y_train_index[-1]
        future_index = pd.date_range(
            start=last_timestamp + pd.Timedelta(minutes=30),
            periods=horizon,
            freq="30min"
        )

        # Create and return the forecast Series
        forecast_series = pd.Series(
            forecast_values.values,
            index=future_index,
            name="Demand"
        )

        return forecast_series

    def save(self, path: str | Path) -> None:
        """
        Save the fitted model to disk using joblib.

        Parameters
        ----------
        path : str | Path
            File path where the model should be saved.
        """
        if self.fitted_model is None:
            raise ValueError("Model has not been fitted. Nothing to save.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "fitted_model": self.fitted_model,
            "y_train_index": self.y_train_index,
            "seasonal_periods": self.seasonal_periods,
            "trend": self.trend,
            "seasonal": self.seasonal,
            "damped_trend": self.damped_trend,
        }
        joblib.dump(model_data, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "HoltWintersForecaster":
        """
        Load a fitted model from disk.

        Parameters
        ----------
        path : str | Path
            File path to the saved model.

        Returns
        -------
        HoltWintersForecaster
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

        # Reconstruct the forecaster
        forecaster = cls(
            seasonal_periods=model_data["seasonal_periods"],
            trend=model_data["trend"],
            seasonal=model_data["seasonal"],
            damped_trend=model_data.get("damped_trend", True),
        )
        forecaster.fitted_model = model_data["fitted_model"]
        forecaster.y_train_index = model_data["y_train_index"]

        return forecaster
