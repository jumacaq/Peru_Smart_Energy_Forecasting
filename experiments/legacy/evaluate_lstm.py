"""
Benchmark LSTMForecaster against HoltWinters, Prophet, LightGBM and their 0.6/0.4 ensemble.

Both horizons use the same splits as the other evaluation scripts: train on everything
before the last `horizon` steps, test on those last steps. All models are fit in this
single run so every number in the table comes from the same code and data.
"""
import random
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

from src.models.statistical import HoltWintersForecaster
from src.models.prophet import ProphetForecaster
from src.models.gbdt import LightGBMForecaster
from src.models.lstm_model import LSTMForecaster
from src.metrics import evaluate_all, evaluation_table

SEED = 42
LSTM_EPOCH_CAP = 20  # early stopping (patience=5) may end training sooner
HORIZONS = {"7-day (336 steps)": 336, "30-day (1,440 steps)": 1440}
OUT_CSV = "docs/benchmark_results.csv"


def set_seeds() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


demand = pd.read_parquet("data/processed/demand.parquet")["Demand"]
print(f"Loaded {len(demand)} points: {demand.index[0]} to {demand.index[-1]}")

rows = []
lstm_info = {}

for horizon_name, horizon in HORIZONS.items():
    print(f"\n{'=' * 80}\n{horizon_name}\n{'=' * 80}")
    train = demand.iloc[:-horizon]
    test = demand.iloc[-horizon:]
    print(f"Train: {len(train)} steps ({train.index[0]} to {train.index[-1]})")
    print(f"Test:  {len(test)} steps ({test.index[0]} to {test.index[-1]})")

    preds = {}
    fit_times = {}

    set_seeds()
    hw = HoltWintersForecaster(seasonal_periods=48, trend="add", seasonal="add", damped_trend=True)
    t0 = time.perf_counter()
    hw.fit(train)
    fit_times["HoltWinters"] = time.perf_counter() - t0
    preds["HoltWinters"] = hw.predict(horizon=horizon)

    set_seeds()
    prophet = ProphetForecaster(
        include_holidays=True, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True
    )
    t0 = time.perf_counter()
    prophet.fit(train)
    fit_times["Prophet"] = time.perf_counter() - t0
    preds["Prophet"] = prophet.predict(horizon=horizon)

    set_seeds()
    gbdt = LightGBMForecaster(num_leaves=31, learning_rate=0.05, n_estimators=100)
    t0 = time.perf_counter()
    gbdt.fit(train)
    fit_times["LightGBM"] = time.perf_counter() - t0
    preds["LightGBM"] = gbdt.predict(horizon=horizon)

    ensemble_values = 0.6 * preds["LightGBM"].values + 0.4 * preds["Prophet"].values
    preds["Ensemble 0.6 LGBM + 0.4 Prophet"] = pd.Series(
        ensemble_values, index=preds["LightGBM"].index, name="Demand"
    )
    fit_times["Ensemble"] = fit_times["LightGBM"] + fit_times["Prophet"]

    set_seeds()
    lstm = LSTMForecaster(epochs=LSTM_EPOCH_CAP, seed=SEED)
    t0 = time.perf_counter()
    lstm.fit(train)
    fit_times["LSTM"] = time.perf_counter() - t0
    preds["LSTM"] = lstm.predict(horizon=horizon)
    lstm_info[horizon_name] = {
        "fit_time_s": fit_times["LSTM"],
        "epochs_run": lstm.epochs_run_,
        "best_val_loss": lstm.best_val_loss_,
    }
    print(
        f"LSTM: fit {fit_times['LSTM']:.1f}s, epochs run {lstm.epochs_run_}/{LSTM_EPOCH_CAP}, "
        f"best_val_loss_ {lstm.best_val_loss_:.6f} (scaled units)"
    )

    for model, pred in preds.items():
        metrics = evaluate_all(test, pred)
        rows.append({
            "horizon": horizon_name,
            "model": model,
            **metrics,
            "fit_time_s": fit_times.get(model.split()[0] if model.startswith("Ensemble") else model),
        })
        print(f"{model:<32} MAE {metrics['MAE']:>10.4f}  RMSE {metrics['RMSE']:>10.4f}  "
              f"MAPE {metrics['MAPE']:>8.4f}%  sMAPE {metrics['sMAPE']:>8.4f}%")

results = pd.DataFrame(rows)
results.to_csv(OUT_CSV, index=False)
print(f"\nSaved {OUT_CSV}")

for horizon_name in HORIZONS:
    subset = results[results["horizon"] == horizon_name].set_index("model")
    table = evaluation_table(
        {m: subset.loc[m, ["MAE", "RMSE", "MAPE", "sMAPE"]].to_dict() for m in subset.index},
        sort_by="RMSE",
    )
    print(f"\n{horizon_name}")
    print(table.round(4).to_string())

print("\nLSTM training details")
for horizon_name, info in lstm_info.items():
    print(f"{horizon_name}: fit {info['fit_time_s']:.1f}s, epochs run {info['epochs_run']}, "
          f"best_val_loss_ {info['best_val_loss']:.6f}")
