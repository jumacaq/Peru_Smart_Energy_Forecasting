"""
Dual-horizon end-to-end evaluation script.
Tests forecasters on both 336-step (7-day) and 1,440-step (30-day) horizons.
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from src.models.statistical import HoltWintersForecaster
from src.models.prophet import ProphetForecaster
from src.metrics import evaluate_all

print("=" * 90)
print("DUAL-HORIZON BENCHMARK: HoltWinters vs Prophet")
print("=" * 90)

# Load the data
print("\nLoading data from data/processed/demand.parquet...")
demand_df = pd.read_parquet("data/processed/demand.parquet")
demand = demand_df.iloc[:, 0]
print(f"Loaded {len(demand)} data points")
print(f"Date range: {demand.index[0]} to {demand.index[-1]}")

# Define horizons
horizons = {
    "7-day (336 steps)": 336,
    "30-day (1,440 steps)": 1440
}

# Store results
results = {}

for horizon_name, horizon in horizons.items():
    print(f"\n{'=' * 90}")
    print(f"EVALUATING: {horizon_name} HORIZON ({horizon} steps)")
    print(f"{'=' * 90}")

    # Split into train/test
    train_data = demand.iloc[:-horizon]
    test_data = demand.iloc[-horizon:]

    print(f"\nTrain data: {len(train_data)} steps ({train_data.index[0]} to {train_data.index[-1]})")
    print(f"Test data: {len(test_data)} steps ({test_data.index[0]} to {test_data.index[-1]})")

    results[horizon_name] = {}

    # ========== HoltWinters ==========
    print(f"\n{'-' * 90}")
    print("HoltWintersForecaster (damped_trend=True)")
    print(f"{'-' * 90}")

    try:
        forecaster_hw = HoltWintersForecaster(
            seasonal_periods=48,
            trend="add",
            seasonal="add",
            damped_trend=True
        )

        print(f"Fitting on {len(train_data)} steps...")
        forecaster_hw.fit(train_data)

        print(f"Predicting {horizon} steps ahead...")
        predictions_hw = forecaster_hw.predict(horizon=horizon)

        metrics_hw = evaluate_all(test_data, predictions_hw)

        print(f"MAE:  {metrics_hw['MAE']:.4f} MW")
        print(f"RMSE: {metrics_hw['RMSE']:.4f} MW")
        print(f"MAPE: {metrics_hw['MAPE']:.4f}%")
        print(f"sMAPE: {metrics_hw['sMAPE']:.4f}%")

        results[horizon_name]['HoltWinters'] = metrics_hw
    except Exception as e:
        print(f"ERROR: {str(e)}")
        results[horizon_name]['HoltWinters'] = None

    # ========== Prophet ==========
    print(f"\n{'-' * 90}")
    print("ProphetForecaster (with Peruvian holidays)")
    print(f"{'-' * 90}")

    try:
        forecaster_prophet = ProphetForecaster(
            include_holidays=True,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True
        )

        print(f"Fitting on {len(train_data)} steps...")
        forecaster_prophet.fit(train_data)

        print(f"Predicting {horizon} steps ahead...")
        predictions_prophet = forecaster_prophet.predict(horizon=horizon)

        metrics_prophet = evaluate_all(test_data, predictions_prophet)

        print(f"MAE:  {metrics_prophet['MAE']:.4f} MW")
        print(f"RMSE: {metrics_prophet['RMSE']:.4f} MW")
        print(f"MAPE: {metrics_prophet['MAPE']:.4f}%")
        print(f"sMAPE: {metrics_prophet['sMAPE']:.4f}%")

        results[horizon_name]['Prophet'] = metrics_prophet
    except Exception as e:
        print(f"ERROR: {str(e)}")
        results[horizon_name]['Prophet'] = None

# ========== Summary Table ==========
print(f"\n\n{'=' * 90}")
print("BENCHMARK SUMMARY TABLE")
print(f"{'=' * 90}\n")

print("7-DAY HORIZON (336 steps)")
print("-" * 90)
print(f"{'Model':<25} {'MAE':<15} {'RMSE':<15} {'MAPE':<15} {'sMAPE':<15}")
print("-" * 90)

for model, metrics in results["7-day (336 steps)"].items():
    if metrics:
        print(f"{model:<25} {metrics['MAE']:<15.4f} {metrics['RMSE']:<15.4f} {metrics['MAPE']:<15.4f} {metrics['sMAPE']:<15.4f}")

print("\n30-DAY HORIZON (1,440 steps)")
print("-" * 90)
print(f"{'Model':<25} {'MAE':<15} {'RMSE':<15} {'MAPE':<15} {'sMAPE':<15}")
print("-" * 90)

for model, metrics in results["30-day (1,440 steps)"].items():
    if metrics:
        print(f"{model:<25} {metrics['MAE']:<15.4f} {metrics['RMSE']:<15.4f} {metrics['MAPE']:<15.4f} {metrics['sMAPE']:<15.4f}")

print(f"\n{'=' * 90}\n")
