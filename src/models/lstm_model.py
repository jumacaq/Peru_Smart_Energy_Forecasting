import copy
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.feature_engineering import add_calendar_features, add_cyclical_encodings
from src.models.base import BaseForecaster

FREQ = "30min"
VAL_FRACTION = 0.1
CALENDAR_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "is_weekend",
    "is_holiday",
]


def _calendar_matrix(index: pd.DatetimeIndex) -> np.ndarray:
    """Calendar features for each timestamp as a float32 array of shape (len(index), 6)."""
    df = add_calendar_features(pd.DataFrame(index=index))
    df = add_cyclical_encodings(df)
    return df[CALENDAR_COLUMNS].to_numpy(dtype=np.float32)


def _select_device() -> torch.device:
    """Pick CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _WindowDataset(Dataset):
    """Lazily slices (lookback window, future calendar, target block) samples from arrays."""

    def __init__(self, demand: np.ndarray, calendar: np.ndarray, starts: np.ndarray,
                 lookback: int, block_size: int):
        self.demand = demand
        self.calendar = calendar
        self.starts = starts
        self.lookback = lookback
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, k: int):
        i = self.starts[k]
        end = i + self.lookback
        x = np.concatenate([self.demand[i:end, None], self.calendar[i:end]], axis=1)
        future_calendar = self.calendar[end:end + self.block_size].reshape(-1)
        y = self.demand[end:end + self.block_size]
        return torch.from_numpy(x), torch.from_numpy(future_calendar), torch.from_numpy(y)


class _LSTMNet(nn.Module):
    """LSTM encoder over the lookback window and a direct head for one block of steps."""

    def __init__(self, n_inputs: int, hidden_dim: int, num_layers: int, dropout: float,
                 block_size: int, n_calendar: int):
        super().__init__()
        self.lstm = nn.LSTM(
            n_inputs,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # The head also sees the calendar features of the steps it predicts.
        self.head = nn.Linear(hidden_dim + block_size * n_calendar, block_size)

    def forward(self, x: torch.Tensor, future_calendar: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        h = self.dropout(h_n[-1])
        return self.head(torch.cat([h, future_calendar], dim=1))


class LSTMForecaster(BaseForecaster):
    """
    PyTorch LSTM forecaster with a direct multi-step block head.

    Each sample is a window of the last `lookback` scaled demand values plus calendar
    features. The network predicts `block_size` future steps at once. Longer horizons
    are reached by rolling forward block by block, feeding predictions back into the
    window, then trimming to the requested horizon.
    """

    def __init__(
        self,
        lookback: int = 336,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        epochs: int = 20,
        batch_size: int = 64,
        lr: float = 1e-3,
        block_size: int = 48,
        seed: int = 42,
        patience: int = 5,
    ):
        """
        Initialize the LSTM forecaster.

        Parameters
        ----------
        lookback : int
            Number of past 30-minute steps fed to the LSTM. Default 336 (one week).
        hidden_dim : int
            LSTM hidden size. Default 64.
        num_layers : int
            Number of LSTM layers, 1 or 2. Default 1.
        dropout : float
            Dropout before the head (and between layers when num_layers=2).
        epochs : int
            Maximum training epochs. Default 20.
        batch_size : int
            Training batch size. Default 64.
        lr : float
            Adam learning rate. Default 1e-3.
        block_size : int
            Number of steps predicted per forward pass. Default 48 (24 hours).
        seed : int
            Seed for Python, NumPy and PyTorch RNGs. Default 42.
        patience : int
            Epochs without validation improvement before early stopping. Default 5.
        """
        if num_layers not in (1, 2):
            raise ValueError("num_layers must be 1 or 2")
        if lookback <= 0 or block_size <= 0:
            raise ValueError("lookback and block_size must be positive integers")

        self.lookback = lookback
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.block_size = block_size
        self.seed = seed
        self.patience = patience

        self.device = _select_device()
        self.net = None
        self.scaler = None
        self.y_train_index = None
        self.last_values = None
        self.best_val_loss_ = None
        self.epochs_run_ = None

    def _hyperparameters(self) -> dict:
        return {
            "lookback": self.lookback,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "block_size": self.block_size,
            "seed": self.seed,
            "patience": self.patience,
        }

    def _build_net(self) -> _LSTMNet:
        return _LSTMNet(
            n_inputs=1 + len(CALENDAR_COLUMNS),
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            block_size=self.block_size,
            n_calendar=len(CALENDAR_COLUMNS),
        ).to(self.device)

    def _set_seeds(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def _run_epoch(self, loader: DataLoader, loss_fn: nn.Module, optimizer=None) -> float:
        """Run one pass over `loader`. Trains when an optimizer is given, otherwise evaluates."""
        training = optimizer is not None
        self.net.train(training)
        total, count = 0.0, 0
        with torch.set_grad_enabled(training):
            for x, future_calendar, y in loader:
                x = x.to(self.device)
                future_calendar = future_calendar.to(self.device)
                y = y.to(self.device)
                pred = self.net(x, future_calendar)
                loss = loss_fn(pred, y)
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                    optimizer.step()
                total += loss.item() * len(y)
                count += len(y)
        return total / count

    def fit(self, y_train: pd.Series, X_train: pd.DataFrame | None = None) -> "LSTMForecaster":
        """
        Fit the LSTM on scaled demand and calendar features.

        The last 10% of the training series is held out for early stopping. Training
        windows only use targets before that split, so validation data never leaks into
        training. The scaler is fit on y_train only.

        Parameters
        ----------
        y_train : pd.Series
            Historical demand with DatetimeIndex.
        X_train : pd.DataFrame | None
            Optional exogenous features (ignored - calendar features are generated internally).

        Returns
        -------
        LSTMForecaster
            Self for method chaining.

        Raises
        ------
        ValueError
            If y_train lacks a DatetimeIndex, has no more than `lookback` points, or is
            too short to form a single training window.
        """
        if not isinstance(y_train.index, pd.DatetimeIndex):
            raise ValueError("y_train must have a DatetimeIndex")
        if len(y_train) <= self.lookback:
            raise ValueError(
                f"y_train must have more than lookback={self.lookback} points, got {len(y_train)}"
            )

        self._set_seeds()

        values = y_train.to_numpy(dtype=np.float64)
        self.scaler = StandardScaler().fit(values.reshape(-1, 1))
        demand = self.scaler.transform(values.reshape(-1, 1)).ravel().astype(np.float32)
        calendar = _calendar_matrix(y_train.index)

        n = len(demand)
        all_starts = np.arange(0, n - self.lookback - self.block_size + 1)
        split = int(n * (1 - VAL_FRACTION))
        train_starts = all_starts[all_starts + self.lookback + self.block_size <= split]
        val_starts = all_starts[all_starts + self.lookback >= split]
        if len(train_starts) == 0:
            raise ValueError(
                "y_train is too short for one training window; "
                f"need at least {math.ceil((self.lookback + self.block_size) / (1 - VAL_FRACTION))} points"
            )

        self.net = self._build_net()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        generator = torch.Generator().manual_seed(self.seed)
        train_loader = DataLoader(
            _WindowDataset(demand, calendar, train_starts, self.lookback, self.block_size),
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )
        val_loader = None
        if len(val_starts) > 0:
            val_loader = DataLoader(
                _WindowDataset(demand, calendar, val_starts, self.lookback, self.block_size),
                batch_size=self.batch_size,
            )
        # Without a validation slice, early stopping falls back to training loss.
        monitor = val_loader if val_loader is not None else train_loader

        best_loss, best_state, epochs_without_gain = math.inf, None, 0
        self.epochs_run_ = 0
        for _ in range(self.epochs):
            self._run_epoch(train_loader, loss_fn, optimizer)
            loss = self._run_epoch(monitor, loss_fn)
            self.epochs_run_ += 1
            if loss < best_loss:
                best_loss = loss
                best_state = copy.deepcopy(self.net.state_dict())
                epochs_without_gain = 0
            else:
                epochs_without_gain += 1
                if epochs_without_gain >= self.patience:
                    break

        if best_state is not None:  # stays None only if every epoch's loss was NaN
            self.net.load_state_dict(best_state)
        self.net.eval()
        self.best_val_loss_ = best_loss
        self.y_train_index = y_train.index
        self.last_values = values[-self.lookback:].copy()

        return self

    def predict(self, horizon: int, X_future: pd.DataFrame | None = None) -> pd.Series:
        """
        Forecast future demand by rolling the block head forward.

        Parameters
        ----------
        horizon : int
            Number of 30-minute steps to forecast. Must be a positive integer.
        X_future : pd.DataFrame | None
            Optional exogenous features (ignored - calendar features come from the index).

        Returns
        -------
        pd.Series
            Forecasts in original units with a future 30-minute DatetimeIndex.

        Raises
        ------
        ValueError
            If the model has not been fitted or horizon <= 0.
        """
        if self.net is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        if horizon <= 0:
            raise ValueError("horizon must be a positive integer")

        n_blocks = math.ceil(horizon / self.block_size)
        future_index = pd.date_range(
            start=self.y_train_index[-1] + pd.Timedelta(FREQ),
            periods=n_blocks * self.block_size,
            freq=FREQ,
        )
        # Calendar features for the whole timeline: lookback history, then the future.
        calendar = np.vstack([
            _calendar_matrix(self.y_train_index[-self.lookback:]),
            _calendar_matrix(future_index),
        ])

        demand = self.scaler.transform(self.last_values.reshape(-1, 1)).ravel().astype(np.float32)
        blocks = []
        self.net.eval()
        with torch.no_grad():
            for b in range(n_blocks):
                start = b * self.block_size
                x = np.concatenate([demand[-self.lookback:, None], calendar[start:start + self.lookback]], axis=1)
                future_calendar = calendar[start + self.lookback:start + self.lookback + self.block_size].reshape(1, -1)
                x_t = torch.from_numpy(x[None]).to(self.device)
                fc_t = torch.from_numpy(future_calendar).to(self.device)
                block = self.net(x_t, fc_t).cpu().numpy().ravel().astype(np.float32)
                blocks.append(block)
                demand = np.concatenate([demand, block])

        scaled = np.concatenate(blocks)[:horizon]
        values = self.scaler.inverse_transform(scaled.reshape(-1, 1)).ravel()
        return pd.Series(values, index=future_index[:horizon], name="Demand")

    def save(self, path: str | Path) -> None:
        """
        Save the fitted model, scaler, hyperparameters and training context to one file.

        Parameters
        ----------
        path : str | Path
            File path where the model should be saved.

        Raises
        ------
        ValueError
            If the model has not been fitted.
        """
        if self.net is None:
            raise ValueError("Model has not been fitted. Nothing to save.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "state_dict": {k: v.detach().cpu() for k, v in self.net.state_dict().items()},
                "scaler": self.scaler,
                "hyperparameters": self._hyperparameters(),
                "y_train_index": self.y_train_index,
                "last_values": self.last_values,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "LSTMForecaster":
        """
        Load a model saved with save(), mapping tensors onto the available device.

        Parameters
        ----------
        path : str | Path
            File path to the saved model.

        Returns
        -------
        LSTMForecaster
            Deserialized model ready for prediction.

        Raises
        ------
        FileNotFoundError
            If the model file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")

        # weights_only=False is needed because the artifact stores a fitted sklearn scaler
        # and a pandas index. Only load files you trust.
        data = torch.load(path, map_location=_select_device(), weights_only=False)

        forecaster = cls(**data["hyperparameters"])
        forecaster.scaler = data["scaler"]
        forecaster.y_train_index = data["y_train_index"]
        forecaster.last_values = data["last_values"]
        forecaster.net = forecaster._build_net()
        forecaster.net.load_state_dict(data["state_dict"])
        forecaster.net.eval()

        return forecaster
