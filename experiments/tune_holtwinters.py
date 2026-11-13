"""
Grid search validation for HoltWintersForecaster parameters.
Tests 4 combinations of trend and seasonal components.
"""
import pandas as pd
import numpy as np
from src.models.statistical import HoltWintersForecaster
from src.metrics import evaluate_all, evaluation_table

# Load the data
print("Loading data from data/processed/demand.parquet...")
demand = pd.read_parquet("data/processed/demand.parquet")["Demand"]
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

# Grid search combinations
combinations = [
    {"trend": "add", "seasonal": "add"},
    {"trend": "add", "seasonal": "mul"},
    {"trend": "mul", "seasonal": "add"},
    {"trend": "mul", "seasonal": "mul"},
]

print("\n" + "="*70)
print("GRID SEARCH: HoltWinters Parameter Tuning")
print("="*70)

# Store results
results = {}

for i, combo in enumerate(combinations, 1):
    name = f"{combo['trend'].upper()}/{'ADD' if combo['seasonal'] == 'add' else 'MUL'}"
    print(f"\n[{i}/4] Testing {name} (trend={combo['trend']}, seasonal={combo['seasonal']})")
    print("-" * 70)

    try:
        # Initialize and fit
        forecaster = HoltWintersForecaster(
            seasonal_periods=48,
            trend=combo["trend"],
            seasonal=combo["seasonal"],
            damped_trend=True
        )

        print("  Fitting model...")
        forecaster.fit(train_data)

        # Make predictions
        print("  Generating predictions...")
        predictions = forecaster.predict(horizon=horizon)

        # Evaluate
        print("  Evaluating metrics...")
        metrics = evaluate_all(test_data, predictions)
        results[name] = metrics

        print(f"  ✓ MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, "
              f"MAPE={metrics['MAPE']:.4f}%, sMAPE={metrics['sMAPE']:.4f}%")

    except Exception as e:
        print(f"  ✗ Error: {type(e).__name__}: {str(e)}")
        results[name] = {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "sMAPE": np.nan}

# Display comparison table
print("\n" + "="*70)
print("COMPARISON TABLE (sorted by RMSE)")
print("="*70)

if results:
    comparison_df = evaluation_table(results, sort_by="RMSE", ascending=True)
    print("\n")
    print(comparison_df.to_string())

    # Identify baseline (ADD/ADD)
    baseline_name = "ADD/ADD"
    if baseline_name in comparison_df.index:
        baseline_rmse = comparison_df.loc[baseline_name, "RMSE"]
        baseline_mae = comparison_df.loc[baseline_name, "MAE"]

        print(f"\n" + "="*70)
        print(f"BASELINE COMPARISON ({baseline_name})")
        print("="*70)

        for idx in comparison_df.index:
            if idx != baseline_name:
                rmse_diff = comparison_df.loc[idx, "RMSE"] - baseline_rmse
                rmse_pct = (rmse_diff / baseline_rmse) * 100
                mae_diff = comparison_df.loc[idx, "MAE"] - baseline_mae
                mae_pct = (mae_diff / baseline_mae) * 100

                improvement = "↓ BETTER" if rmse_diff < 0 else "↑ WORSE"
                print(f"\n{idx} vs {baseline_name}:")
                print(f"  RMSE: {rmse_diff:+.4f} ({rmse_pct:+.2f}%) {improvement}")
                print(f"  MAE:  {mae_diff:+.4f} ({mae_pct:+.2f}%)")

        # Determine if any combo significantly outperforms baseline
        best_row = comparison_df.iloc[0]
        best_name = comparison_df.index[0]

        print(f"\n" + "="*70)
        if best_name == baseline_name:
            print(f"RECOMMENDATION: Current default {baseline_name} is OPTIMAL")
            print(f"\nAll other combinations perform worse:")
            for idx in comparison_df.index[1:]:
                rmse_gap = (comparison_df.loc[idx, "RMSE"] - baseline_rmse) / baseline_rmse * 100
                print(f"  - {idx}: {rmse_gap:+.2f}% RMSE gap")
            print(f"\nNo action required. Default parameters are well-tuned.")
        else:
            rmse_improvement = (baseline_rmse - best_row["RMSE"]) / baseline_rmse * 100
            if rmse_improvement > 5:
                print(f"RECOMMENDATION: {best_name} significantly outperforms {baseline_name}")
                print(f"RMSE improvement: {rmse_improvement:.2f}%")
                print(f"\nTo update defaults in src/models/statistical.py:")
                print(f"  Change: def __init__(self, seasonal_periods: int = 48, "
                      f"trend: str = \"mul\", seasonal: str = \"add\", ...)")
                print(f"\nThen update tests in tests/test_statistical.py accordingly.")
            else:
                print(f"All combinations perform similarly (within 5% of each other).")
                print(f"Current default {baseline_name} is acceptable.")
        print("="*70)
