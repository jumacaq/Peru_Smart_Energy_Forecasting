from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursive
from src.models.base import BaseForecaster
from src.feature_engineering import add_calendar_features, add_cyclical_encodings


class LightGBMForecaster(BaseForecaster):
    """
    LightGBM (GBDT) forecaster using skforecast's recursive framework with calendar exogenous features.

    Uses lags for autoregressive components and calendar features (hour, day_of_week, month,
    cyclical encodings, holidays) as exogenous variables with skforecast's ForecasterRecursive
    for robust multi-step recursive forecasting. GBDT can better capture nonlinear calendar
    interactions than Prophet's additive components.
    """

    def __init__(self, num_leaves: int = 31, learning_rate: float = 0.05, n_estimators: int = 100):
        """
        Initialize the LightGBM forecaster.

        Parameters
        ----------
        num_leaves : int
            Number of leaves in each tree. Default 31.
        learning_rate : float
            Learning rate (shrinkage). Default 0.05.
        n_estimators : int
            Number of boosting rounds. Default 100.
        """
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.forecaster = None
        self.y_train_index = None

    def _get_calendar_features(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Generate calendar features for given datetime index.

        Parameters
        ----------
        index : pd.DatetimeIndex
            Datetime index to generate features for.

        Returns
        -------
        pd.DataFrame
            DataFrame with calendar features and cyclical encodings.
        """
        df = pd.DataFrame(index=index)
        df = add_calendar_features(df)
        df = add_cyclical_encodings(df)

        # Ensure is_holiday is present (add_calendar_features should include it)
        if 'is_holiday' not in df.columns:
            df['is_holiday'] = 0

        return df

    def fit(self, y_train: pd.Series, X_train: pd.DataFrame | None = None) -> "LightGBMForecaster":
        """
        Fit the LightGBM forecaster with calendar features as exogenous variables.

        Parameters
        ----------
        y_train : pd.Series
            Historical target values with DatetimeIndex.
        X_train : pd.DataFrame | None
            Optional exogenous features (ignored - calendar features generated internally).

        Returns
        -------
        LightGBMForecaster
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

        # Generate calendar features for training period
        exog_train = self._get_calendar_features(y_train.index)

        # Create LGBMRegressor
        estimator = LGBMRegressor(
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            verbose=-1,
            random_state=42
        )

        # Create ForecasterRecursive with skforecast
        # Lags handle autoregressive components; exog handles calendar context
        self.forecaster = ForecasterRecursive(
            estimator=estimator,
            lags=[1, 2, 48, 336]
        )

        # Ensure the Series has a frequency for skforecast
        y_train_with_freq = y_train.copy()
        if y_train_with_freq.index.freq is None:
            y_train_with_freq.index.freq = pd.infer_freq(y_train_with_freq.index)

        # Fit the forecaster with calendar features as exogenous variables
        self.forecaster.fit(y=y_train_with_freq, exog=exog_train)

        return self

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        """
        Forecast future values with calendar context via exogenous features.

        Parameters
        ----------
        horizon : int
            Number of 30-minute steps to forecast.
        X_future : pd.DataFrame | None
            Optional exogenous features (ignored - calendar features generated for horizon).

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
        if self.forecaster is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        if horizon <= 0:
            raise ValueError("horizon must be a positive integer")

        if self.y_train_index is None:
            raise RuntimeError("Training data index not available")

        # Generate future DatetimeIndex
        last_timestamp = self.y_train_index[-1]
        future_index = pd.date_range(
            start=last_timestamp + pd.Timedelta(minutes=30),
            periods=horizon,
            freq="30min"
        )

        # Generate calendar features for forecast horizon
        exog_future = self._get_calendar_features(future_index)

        # Use skforecast's predict method with exogenous features
        predictions = self.forecaster.predict(steps=horizon, exog=exog_future)

        # Create and return the forecast Series
        forecast_series = pd.Series(
            predictions,
            index=future_index,
            name="Demand"
        )

        return forecast_series

    def save(self, path: str | Path) -> None:
        """
        Save the fitted forecaster to disk using joblib.

        Parameters
        ----------
        path : str | Path
            File path where the model should be saved.
        """
        if self.forecaster is None:
            raise ValueError("Model has not been fitted. Nothing to save.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "forecaster": self.forecaster,
            "y_train_index": self.y_train_index,
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
        }
        joblib.dump(model_data, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "LightGBMForecaster":
        """
        Load a fitted forecaster from disk.

        Parameters
        ----------
        path : str | Path
            File path to the saved model.

        Returns
        -------
        LightGBMForecaster
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
            num_leaves=model_data.get("num_leaves", 31),
            learning_rate=model_data.get("learning_rate", 0.05),
            n_estimators=model_data.get("n_estimators", 100),
        )
        forecaster.forecaster = model_data["forecaster"]
        forecaster.y_train_index = model_data["y_train_index"]

        return forecaster
