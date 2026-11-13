from pathlib import Path
import pandas as pd
import joblib
from prophet import Prophet
import holidays
from src.models.base import BaseForecaster


class ProphetForecaster(BaseForecaster):
    """
    Prophet forecaster for time series prediction with holiday effects.

    Supports Peruvian national holidays and handles single seasonality
    (seasonal_periods=48 for 30-minute intervals covering daily patterns).
    """

    def __init__(self, include_holidays: bool = True, yearly_seasonality: bool = True, weekly_seasonality: bool = True, daily_seasonality: bool = True):
        """
        Initialize the Prophet forecaster.

        Parameters
        ----------
        include_holidays : bool
            Whether to include Peruvian national holidays. Default True.
        yearly_seasonality : bool
            Whether to fit yearly seasonality. Default True.
        weekly_seasonality : bool
            Whether to fit weekly seasonality. Default True.
        daily_seasonality : bool
            Whether to fit daily seasonality. Default True.
        """
        self.include_holidays = include_holidays
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.model = None
        self.fitted_model = None
        self.y_train_index = None
        self.holidays_df = None

    def _get_holidays_df(self) -> pd.DataFrame | None:
        """
        Generate a DataFrame of Peruvian national holidays for Prophet.

        Returns
        -------
        pd.DataFrame | None
            DataFrame with columns 'holiday' and 'ds' for Prophet, or None if include_holidays=False.
        """
        if not self.include_holidays:
            return None

        # Get Peruvian holidays for a broad date range
        # This covers the typical training data period
        peru_holidays = holidays.Peru(years=range(2020, 2030))

        # Convert to Prophet-compatible format
        holidays_list = []
        for date, name in sorted(peru_holidays.items()):
            holidays_list.append({
                'holiday': name,
                'ds': pd.Timestamp(date)
            })

        if not holidays_list:
            return None

        holidays_df = pd.DataFrame(holidays_list)
        return holidays_df

    def fit(self, y_train: pd.Series, X_train: pd.DataFrame | None = None) -> "ProphetForecaster":
        """
        Fit the Prophet model.

        Parameters
        ----------
        y_train : pd.Series
            Historical target values with DatetimeIndex.
        X_train : pd.DataFrame | None
            Optional exogenous features (ignored for Prophet).

        Returns
        -------
        ProphetForecaster
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

        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        df = pd.DataFrame({
            'ds': y_train.index,
            'y': y_train.values
        })

        # Get holidays if needed
        self.holidays_df = self._get_holidays_df()

        # Initialize Prophet with specified seasonality parameters
        self.fitted_model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            holidays=self.holidays_df,
            interval_width=0.95
        )

        # Fit the model (fit returns the model itself)
        self.fitted_model.fit(df)

        return self

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        """
        Forecast future values.

        Parameters
        ----------
        horizon : int
            Number of 30-minute steps to forecast.
        X_future : pd.DataFrame | None
            Optional exogenous features (ignored for Prophet).

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

        # Create future DataFrame with 30-minute frequency
        last_timestamp = self.y_train_index[-1]
        future_index = pd.date_range(
            start=last_timestamp + pd.Timedelta(minutes=30),
            periods=horizon,
            freq="30min"
        )

        future_df = pd.DataFrame({'ds': future_index})

        # Make forecast
        forecast = self.fitted_model.predict(future_df)

        # Extract the point forecast (yhat)
        forecast_values = forecast['yhat'].values

        # Create and return the forecast Series
        forecast_series = pd.Series(
            forecast_values,
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
            "include_holidays": self.include_holidays,
            "yearly_seasonality": self.yearly_seasonality,
            "weekly_seasonality": self.weekly_seasonality,
            "daily_seasonality": self.daily_seasonality,
            "holidays_df": self.holidays_df,
        }
        joblib.dump(model_data, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "ProphetForecaster":
        """
        Load a fitted model from disk.

        Parameters
        ----------
        path : str | Path
            File path to the saved model.

        Returns
        -------
        ProphetForecaster
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
            include_holidays=model_data.get("include_holidays", True),
            yearly_seasonality=model_data.get("yearly_seasonality", True),
            weekly_seasonality=model_data.get("weekly_seasonality", True),
            daily_seasonality=model_data.get("daily_seasonality", True),
        )
        forecaster.fitted_model = model_data["fitted_model"]
        forecaster.y_train_index = model_data["y_train_index"]
        forecaster.holidays_df = model_data.get("holidays_df")

        return forecaster
