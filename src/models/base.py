from abc import ABC, abstractmethod
import pandas as pd


class BaseForecaster(ABC):
    """Abstract base class for all forecasting model wrappers."""

    @abstractmethod
    def fit(self, y_train: pd.Series, X_train: pd.DataFrame | None = None) -> "BaseForecaster":
        """
        Fit the forecasting model on historical target data and optional exogenous features.

        Parameters
        ----------
        y_train : pd.Series
            Historical target values with DatetimeIndex. Index must be sorted and represent
            30-minute intervals (or the native frequency of the data).
        X_train : pd.DataFrame | None, optional
            Optional exogenous features with DatetimeIndex matching y_train. Each column
            represents a different feature. Default is None.

        Returns
        -------
        BaseForecaster
            Returns self to allow method chaining.
        """
        ...

    @abstractmethod
    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        """
        Forecast 'horizon' steps ahead into the future.

        Parameters
        ----------
        horizon : int
            Number of 30-minute steps to forecast (e.g., 48 = 24 hours, 336 = 7 days).
            Must be a positive integer.
        X_future : pd.DataFrame | None, optional
            Optional exogenous features for the forecast horizon with DatetimeIndex
            spanning exactly 'horizon' periods. Default is None.

        Returns
        -------
        pd.Series
            Forecasted values with DatetimeIndex. The index must:
            - Start after the last index in the training data
            - Contain exactly 'horizon' datetime points
            - Maintain 30-minute interval frequency consistent with training data
            - Be sorted in ascending order
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Serialize the fitted model artifact to disk.

        Parameters
        ----------
        path : str
            File path where the model should be saved. Parent directories should exist.
            Common convention: use .pkl, .joblib, or .h5 extensions depending on serialization method.
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseForecaster":
        """
        Load a serialized model artifact from disk.

        Parameters
        ----------
        path : str
            File path to the saved model artifact.

        Returns
        -------
        BaseForecaster
            Instantiated and deserialized model object ready for prediction.

        Raises
        ------
        FileNotFoundError
            If the model file does not exist at the given path.
        """
        ...
