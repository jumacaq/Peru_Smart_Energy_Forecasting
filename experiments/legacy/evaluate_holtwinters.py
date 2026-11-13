"""
End-to-end evaluation of HoltWintersForecaster on real demand data.
"""
import pandas as pd
import numpy as np
from src.models.statistical import HoltWintersForecaster
from src.metrics import evaluate_all

# Load the data
print("Loading data from data/processed/demand.parquet...")
demand = pd.read_parquet("data/processed/demand.parquet")
print(f"Loaded {len(demand)} data points")
print(f"Date range: {demand.index[0]} to {demand.index[-1]}")

# Define horizon (7 days = 336 30-minute steps)
horizon = 336
print(f"\nUsing forecast horizon: {horizon} steps (7 days)")

# Split into train/test
# Use last 336 steps as test set
train_data = demand.iloc[:-horizon]
test_data = demand.iloc[-horizon:]

print(f"\nTrain data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")
print(f"Train date range: {train_data.index[0]} to {train_data.index[-1]}")
print(f"Test date range: {test_data.index[0]} to {test_data.index[-1]}")

# Initialize and fit HoltWintersForecaster
print("\nInitializing HoltWintersForecaster with damped_trend=True...")
forecaster = HoltWintersForecaster(
    seasonal_periods=48,  # 30-minute intervals (24 hours)
    trend="add",
    seasonal="add",
    damped_trend=True
)

print("Fitting model on training data...")
forecaster.fit(train_data)

# Make predictions
print(f"Making {horizon}-step forecast...")
predictions = forecaster.predict(horizon=horizon)

print(f"\nPrediction range: {predictions.index[0]} to {predictions.index[-1]}")
print(f"Prediction shape: {predictions.shape}")
print(f"Predictions - Min: {predictions.min():.2f}, Max: {predictions.max():.2f}, Mean: {predictions.mean():.2f}")

# Evaluate
print("\nEvaluating predictions...")
metrics = evaluate_all(test_data, predictions)

print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)
print(f"MAE (Mean Absolute Error):           {metrics['MAE']:.4f}")
print(f"RMSE (Root Mean Squared Error):      {metrics['RMSE']:.4f}")
print(f"MAPE (Mean Absolute % Error):        {metrics['MAPE']:.4f}%")
print(f"sMAPE (Symmetric Mean Absolute % Error): {metrics['sMAPE']:.4f}%")
print("="*60)

# Additional statistics
print("\nAdditional Statistics:")
print(f"Actual demand - Min: {test_data.min().values[0]:.2f}, Max: {test_data.max().values[0]:.2f}, Mean: {test_data.mean().values[0]:.2f}")
print(f"Predictions   - Min: {predictions.min():.2f}, Max: {predictions.max():.2f}, Mean: {predictions.mean():.2f}")

# Residuals
residuals = test_data.values - predictions.values
print(f"\nResiduals - Min: {residuals.min():.2f}, Max: {residuals.max():.2f}, Mean: {residuals.mean():.2f}, Std: {residuals.std():.2f}")
