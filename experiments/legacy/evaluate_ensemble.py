"""
Compute actual ensemble predictions: 0.6 * LightGBM + 0.4 * Prophet
Evaluate on both 7-day and 30-day horizons.
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from src.models.statistical import HoltWintersForecaster
from src.models.prophet import ProphetForecaster
from src.models.gbdt import LightGBMForecaster
from src.metrics import evaluate_all

print("=" * 90)
print("ENSEMBLE EVALUATION: 0.6 * LightGBM + 0.4 * Prophet")
print("=" * 90)

# Load the data
print("\nLoading data...")
demand_df = pd.read_parquet("data/processed/demand.parquet")
demand = demand_df.iloc[:, 0]

# Define horizons
horizons = {
    "7-day (336 steps)": 336,
    "30-day (1,440 steps)": 1440
}

results = {}

for horizon_name, horizon in horizons.items():
    print(f"\n{'=' * 90}")
    print(f"{horizon_name.upper()}")
    print(f"{'=' * 90}")

    # Split into train/test
    train_data = demand.iloc[:-horizon]
    test_data = demand.iloc[-horizon:]

    results[horizon_name] = {}

    # ========== LightGBM ==========
    print(f"\nFitting LightGBM...")
    forecaster_gbdt = LightGBMForecaster(n_estimators=100)
    forecaster_gbdt.fit(train_data)
    predictions_gbdt = forecaster_gbdt.predict(horizon=horizon)
    metrics_gbdt = evaluate_all(test_data, predictions_gbdt)
    results[horizon_name]['LightGBM'] = metrics_gbdt
    print(f"LightGBM MAE: {metrics_gbdt['MAE']:.4f} MW")

    # ========== Prophet ==========
    print(f"Fitting Prophet...")
    forecaster_prophet = ProphetForecaster(include_holidays=True)
    forecaster_prophet.fit(train_data)
    predictions_prophet = forecaster_prophet.predict(horizon=horizon)
    metrics_prophet = evaluate_all(test_data, predictions_prophet)
    results[horizon_name]['Prophet'] = metrics_prophet
    print(f"Prophet MAE: {metrics_prophet['MAE']:.4f} MW")

    # ========== Ensemble ==========
    print(f"Computing ensemble (0.6 * LightGBM + 0.4 * Prophet)...")
    predictions_ensemble = 0.6 * predictions_gbdt.values + 0.4 * predictions_prophet.values
    predictions_ensemble = pd.Series(predictions_ensemble, index=predictions_gbdt.index, name="Demand")
    metrics_ensemble = evaluate_all(test_data, predictions_ensemble)
    results[horizon_name]['Ensemble'] = metrics_ensemble
    print(f"Ensemble MAE: {metrics_ensemble['MAE']:.4f} MW")

    # Print comparison
    print(f"\n{'-' * 90}")
    print(f"{'Metric':<15} {'LightGBM':<18} {'Prophet':<18} {'Ensemble':<18}")
    print(f"{'-' * 90}")
    print(f"{'MAE':<15} {metrics_gbdt['MAE']:<18.4f} {metrics_prophet['MAE']:<18.4f} {metrics_ensemble['MAE']:<18.4f}")
    print(f"{'RMSE':<15} {metrics_gbdt['RMSE']:<18.4f} {metrics_prophet['RMSE']:<18.4f} {metrics_ensemble['RMSE']:<18.4f}")
    print(f"{'MAPE':<15} {metrics_gbdt['MAPE']:<18.4f} {metrics_prophet['MAPE']:<18.4f} {metrics_ensemble['MAPE']:<18.4f}")
    print(f"{'sMAPE':<15} {metrics_gbdt['sMAPE']:<18.4f} {metrics_prophet['sMAPE']:<18.4f} {metrics_ensemble['sMAPE']:<18.4f}")

# ========== Summary ==========
print(f"\n\n{'=' * 90}")
print("ENSEMBLE SUMMARY")
print(f"{'=' * 90}\n")

for horizon_name in horizons.keys():
    print(f"\n{horizon_name}")
    print(f"{'-' * 90}")
    print(f"{'Model':<20} {'MAE':<15} {'RMSE':<15} {'MAPE':<15} {'sMAPE':<15}")
    print(f"{'-' * 90}")

    for model in ['LightGBM', 'Prophet', 'Ensemble']:
        metrics = results[horizon_name][model]
        print(f"{model:<20} {metrics['MAE']:<15.4f} {metrics['RMSE']:<15.4f} {metrics['MAPE']:<15.4f} {metrics['sMAPE']:<15.4f}")

print(f"\n{'=' * 90}\n")
