# Peru Smart Energy Forecasting
 
End-to-end forecasting of Peru's electricity demand at 30-minute resolution, built on public COES data. The project compares four forecasting approaches against a seasonal-naive baseline with a rolling-origin benchmark, saves the selected model, and serves it in a Streamlit app.
 
**Live demo:** [perusmartenergyforecasting.streamlit.app](https://perusmartenergyforecasting.streamlit.app/)
 
The project started as a skforecast notebook comparing gradient-boosting models (XGBoost, LightGBM, CatBoost) with lag and calendar features, and grew into a modular pipeline with a proper benchmark and an app.
 
## Key findings
 
- **No model significantly beats the seasonal-naive baseline** (repeat the same half-hour from last week) at 7 or 30 days, using a paired Wilcoxon test over 12 monthly origins.
- **At 7 days, LSTM has the lowest mean RMSE, but the gap to LightGBM is not significant** (p = 0.470). The deployed model is LightGBM: with a statistical tie, the rule picks the faster model (0.12 s vs 149 s to fit).
- **At 30 days, Prophet has the lowest mean RMSE**, but its gap to LightGBM is also not significant (p = 0.622).
- **Holt-Winters is clearly the weakest** at both horizons.
- Month-to-month variation (std of roughly 75-110 MAE) is larger than most differences between models, so a single backtest window would have given a misleading ranking. This is the reason for the 12-origin design.
## Data
 
- Source: COES (Peru's system operator), demand in MW, 30-minute steps.
- Period: 2024-07-01 00:30 to 2026-07-01 00:00, 35,040 rows, continuous with no missing values after cleaning.
- Cleaning: the series is reindexed to a strict 30-minute grid. Gaps of up to 4 steps are interpolated in time. Longer gaps are left as NaN, never partially filled.
- Processed series: `data/processed/demand.parquet`. All paths and benchmark constants live in `src/config.py`; scripts and tests import them instead of hardcoding paths.
- Features (for the models that use them): lags, shifted rolling statistics, calendar variables, cyclical encodings, and Peruvian public holidays.
## Models
 
| Model | Implementation | Notes |
|---|---|---|
| SeasonalNaive (baseline) | `src/models/baseline.py` | Repeats the last 336 steps (one week) |
| Holt-Winters | statsmodels | Seasonal period 48, damped trend |
| Prophet | Prophet | Needs more than two years of history to fit yearly seasonality well |
| LightGBM | skforecast `ForecasterRecursive` | Lags 1, 2, 48, 336 plus calendar exogenous variables |
| LSTM | PyTorch (CPU) | Lookback 336 steps, calendar inputs, scaler fit on training data only |
 
All models share the interface in `src/models/base.py`: `fit`, `predict(horizon)`, `save`, `load`.
 
## Evaluation
 
Rolling-origin backtest with an expanding window:
 
- 12 monthly origins, from 2025-07-01 to 2026-06-01 (00:00).
- At each origin, every model is refit on all data up to the origin and forecasts the next 1,440 steps (30 days).
- 7-day metrics use the first 336 of those steps. 30-day metrics use all 1,440.
- Seed 42. Metrics are reported as mean ± std across the 12 origins.
- Wins count origins where a model has the lowest RMSE among the non-baseline models. Skill is `1 - RMSE / RMSE_baseline`, averaged over origins.
- Model selection: rank by mean 7-day RMSE, run a paired Wilcoxon signed-rank test on the top two, and choose the faster model if the test is not significant (alpha 0.05).
### Results (mean ± std over 12 origins)
 
**7-day horizon**
 
| Model | MAE | RMSE | Fit time (s) | Wins | Skill vs naive |
|---|---|---|---|---|---|
| LSTM | 178.47 ± 75.04 | 222.54 ± 94.91 | 148.9 | 6 | -0.004 |
| LightGBM | 196.73 ± 73.75 | 259.65 ± 114.10 | 0.12 | 5 | 0.019 |
| Prophet | 214.69 ± 78.92 | 263.97 ± 82.69 | 9.6 | 1 | -0.104 |
| SeasonalNaive | 216.86 ± 106.80 | 289.20 ± 142.85 | 0.00 | n/a | 0.000 |
| Holt-Winters | 328.48 ± 70.84 | 415.29 ± 85.15 | 5.2 | 0 | -0.785 |
 
**30-day horizon**
 
| Model | MAE | RMSE | Fit time (s) | Wins | Skill vs naive |
|---|---|---|---|---|---|
| Prophet | 211.44 ± 38.76 | 268.61 ± 42.66 | 9.6 | 4 | 0.021 |
| LightGBM | 219.33 ± 51.31 | 278.89 ± 63.42 | 0.12 | 3 | 0.005 |
| LSTM | 241.73 ± 110.07 | 290.32 ± 117.99 | 148.9 | 5 | -0.133 |
| SeasonalNaive | 212.11 ± 87.27 | 308.10 ± 127.96 | 0.00 | n/a | 0.000 |
| Holt-Winters | 331.83 ± 90.74 | 416.91 ± 76.86 | 5.2 | 0 | -0.496 |
 
The full per-origin results are in `reports/benchmark_origins.csv`, and the saved forecasts for every origin and model are in `reports/predictions_origins.parquet`.
 
## App
 
Run with `uv run streamlit run app.py`. Three tabs:
 
- **Forecast:** forecast from the saved model for 7 or 30 days, with the typical backtest error for the chosen horizon and a CSV download.
- **Backtest:** pick a past origin and compare the saved model with actual demand.
- **Model comparison:** a verdict generated from the benchmark files (best model, significance, comparison with the baseline), an average table, RMSE by month, and forecast vs actual for any origin and model. It reads the saved results only and does not load torch or Prophet.
The forecast starts at the end of the data (2026-07-01 00:30), not from today's date.
 
## Quick start
 
Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).
 
```bash
uv sync
uv run streamlit run app.py          # app, uses the saved model in artifacts/
uv run pytest -q                     # test suite
uv run python scripts/run_benchmark.py                 # full benchmark, about 30-45 min (LSTM is the slow part)
uv run python scripts/run_benchmark.py --skip-lstm     # faster run without the LSTM
```
 
Useful benchmark flags: `--n-origins N`, `--output-dir PATH`, `--no-artifacts` (do not overwrite the saved model), `--resume`.
 
## Project structure
 
```
app.py                      Streamlit app
scripts/run_benchmark.py    Rolling-origin benchmark, winner selection, artifact export
src/
  config.py                 Paths and benchmark constants
  data_loader.py            Load, clean and validate the COES series
  feature_engineering.py    Lags, rolling stats, calendar, holidays
  metrics.py                MAE, RMSE, MAPE, sMAPE
  evaluation.py             Origins, scoring, summary, winner selection
  comparison.py             Verdict and tables for the app
  models/                   Forecasters (base, baseline, statistical, prophet, gbdt, lstm_model)
artifacts/                  Saved best model and its metadata
reports/                    Benchmark results and predictions
tests/                      pytest suite
experiments/                One-off tuning scripts
```
 
## Limitations
 
- **Short history.** About two years of data. Prophet warns that yearly seasonality is under-identified, and no model has seen a full set of yearly patterns more than twice.
- **Small sample of origins.** With 12 origins, the significance tests have low power, and p-values are not corrected for multiple comparisons. "Not significant" means the data cannot separate the models, not that they are equal.
- **Error accumulation.** Recursive models drift over long horizons, and the 30-day forecast is not reliably better than repeating last week's pattern.
- **No exogenous drivers** beyond calendar and holidays (no weather, economic activity or outages), and no prediction intervals.
- **Forecast origin.** The saved model was refit on all data up to 2026-07-01, so the app's forecast covers July 2026.
## Possible next steps
 
- Extend the data with later COES releases and evaluate the July 2026 forecast against what actually happened.
- Add weather variables and prediction intervals.
- Tune and re-test the models once more history is available.
## Data source and license
 
Demand data comes from [COES](https://www.coes.org.pe/Portal/portalinformacion/demandaq). Check COES terms before redistributing the raw data.