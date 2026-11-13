"""
Comprehensive dual-horizon evaluation: HoltWinters, Prophet, and LightGBM.
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from src.models.statistical import HoltWintersForecaster
from src.models.prophet import ProphetForecaster
from src.models.gbdt import LightGBMForecaster
from src.metrics import evaluate_all

print("=" * 110)
print("COMPREHENSIVE BENCHMARK: HoltWinters vs Prophet vs LightGBM")
print("=" * 110)

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
    print(f"\n{'=' * 110}")
    print(f"EVALUATING: {horizon_name} HORIZON ({horizon} steps)")
    print(f"{'=' * 110}")

    # Split into train/test
    train_data = demand.iloc[:-horizon]
    test_data = demand.iloc[-horizon:]

    print(f"\nTrain data: {len(train_data)} steps ({train_data.index[0]} to {train_data.index[-1]})")
    print(f"Test data: {len(test_data)} steps ({test_data.index[0]} to {test_data.index[-1]})")

    results[horizon_name] = {}

    # ========== HoltWinters ==========
    print(f"\n{'-' * 110}")
    print("HoltWintersForecaster (damped_trend=True)")
    print(f"{'-' * 110}")

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

        print(f"✓ MAE:  {metrics_hw['MAE']:.4f} MW")
        print(f"✓ RMSE: {metrics_hw['RMSE']:.4f} MW")
        print(f"✓ MAPE: {metrics_hw['MAPE']:.4f}%")
        print(f"✓ sMAPE: {metrics_hw['sMAPE']:.4f}%")

        results[horizon_name]['HoltWinters'] = metrics_hw
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
        results[horizon_name]['HoltWinters'] = None

    # ========== Prophet ==========
    print(f"\n{'-' * 110}")
    print("ProphetForecaster (with Peruvian holidays)")
    print(f"{'-' * 110}")

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

        print(f"✓ MAE:  {metrics_prophet['MAE']:.4f} MW")
        print(f"✓ RMSE: {metrics_prophet['RMSE']:.4f} MW")
        print(f"✓ MAPE: {metrics_prophet['MAPE']:.4f}%")
        print(f"✓ sMAPE: {metrics_prophet['sMAPE']:.4f}%")

        results[horizon_name]['Prophet'] = metrics_prophet
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
        results[horizon_name]['Prophet'] = None

    # ========== LightGBM ==========
    print(f"\n{'-' * 110}")
    print("LightGBMForecaster (with engineered features)")
    print(f"{'-' * 110}")

    try:
        forecaster_gbdt = LightGBMForecaster(
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=100
        )

        print(f"Fitting on {len(train_data)} steps...")
        forecaster_gbdt.fit(train_data)

        print(f"Predicting {horizon} steps ahead...")
        predictions_gbdt = forecaster_gbdt.predict(horizon=horizon)

        metrics_gbdt = evaluate_all(test_data, predictions_gbdt)

        print(f"✓ MAE:  {metrics_gbdt['MAE']:.4f} MW")
        print(f"✓ RMSE: {metrics_gbdt['RMSE']:.4f} MW")
        print(f"✓ MAPE: {metrics_gbdt['MAPE']:.4f}%")
        print(f"✓ sMAPE: {metrics_gbdt['sMAPE']:.4f}%")

        results[horizon_name]['LightGBM'] = metrics_gbdt
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
        results[horizon_name]['LightGBM'] = None

# ========== Summary Tables ==========
print(f"\n\n{'=' * 110}")
print("BENCHMARK SUMMARY TABLES")
print(f"{'=' * 110}\n")

# Create a formatted table for both horizons
for horizon_name in horizons.keys():
    print(f"\n{horizon_name.upper()}")
    print("-" * 110)
    print(f"{'Model':<20} {'MAE':<18} {'RMSE':<18} {'MAPE':<18} {'sMAPE':<18}")
    print("-" * 110)

    for model, metrics in results[horizon_name].items():
        if metrics:
            print(f"{model:<20} {metrics['MAE']:<18.4f} {metrics['RMSE']:<18.4f} {metrics['MAPE']:<18.4f} {metrics['sMAPE']:<18.4f}")
        else:
            print(f"{model:<20} {'ERROR':<18} {'ERROR':<18} {'ERROR':<18} {'ERROR':<18}")

# ========== Consolidated Comparison Table ==========
print(f"\n\n{'=' * 110}")
print("CONSOLIDATED COMPARISON TABLE")
print(f"{'=' * 110}\n")

models = ['HoltWinters', 'Prophet', 'LightGBM']
metrics_names = ['MAE', 'RMSE', 'MAPE', 'sMAPE']

for metric in metrics_names:
    print(f"\n{metric} COMPARISON ACROSS HORIZONS")
    print("-" * 70)
    print(f"{'Model':<20} {'7-day (336 steps)':<25} {'30-day (1,440 steps)':<25}")
    print("-" * 70)

    for model in models:
        val_7d = results["7-day (336 steps)"].get(model, {}).get(metric, 'N/A') if results["7-day (336 steps)"].get(model) else 'ERROR'
        val_30d = results["30-day (1,440 steps)"].get(model, {}).get(metric, 'N/A') if results["30-day (1,440 steps)"].get(model) else 'ERROR'

        if isinstance(val_7d, float) and isinstance(val_30d, float):
            print(f"{model:<20} {val_7d:<25.4f} {val_30d:<25.4f}")
        else:
            print(f"{model:<20} {str(val_7d):<25} {str(val_30d):<25}")

print(f"\n{'=' * 110}\n")
