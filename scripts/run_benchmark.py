"""
Rolling-origin benchmark of all registered forecasters, with winner selection.

Protocol
--------
* Origins: N_ORIGINS monthly origins at 00:00 on the first day of each month, starting
  at FIRST_ORIGIN (see src/evaluation.py). For each origin and model the model is fit
  once on all data up to and including the origin, then predicts LONG_HORIZON (1,440)
  steps. Scores are computed for the first SHORT_HORIZON (336) steps (7 days, "7d") and
  for all 1,440 steps (30 days, "30d").
* Selection: src.evaluation.select_winner ranks the non-baseline models by mean 7-day
  RMSE across origins, with a paired Wilcoxon test on the top two.
* Artifacts: the winner is refit on the full dataset and saved through its own save()
  to artifacts/best_forecaster.pkl, with metadata in artifacts/best_forecaster_meta.json.
  Both are written to temporary files first and only replace the existing artifacts
  after the whole run has succeeded.

Outputs (in --output-dir, default reports/)
-------------------------------------------
* benchmark_origins.csv        one row per origin, model and horizon (written after each origin)
* predictions_origins.parquet  long format: origin, model, timestamp, step, y_true, y_pred
* benchmark_results.csv        output of src.evaluation.summarize

Usage
-----
    python scripts/run_benchmark.py                          # all models, all origins
    python scripts/run_benchmark.py --skip-lstm              # skip the (slow) LSTM
    python scripts/run_benchmark.py --n-origins 3 --output-dir /tmp/bench --no-artifacts
    python scripts/run_benchmark.py --resume                 # continue an interrupted run
"""

import argparse
import json
import logging
import os
import random
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    ARTIFACTS_DIR,
    BASELINE_NAME,
    BENCHMARK_ORIGINS_CSV,
    BENCHMARK_RESULTS_CSV,
    BEST_MODEL_META_PATH,
    BEST_MODEL_PATH,
    DEMAND_PARQUET_PATH,
    LONG_HORIZON,
    N_ORIGINS,
    PREDICTIONS_PARQUET,
    REPORTS_DIR,
    ROOT,
    SEED,
    SHORT_HORIZON,
)
from src.data_loader import load_processed_dataset
from src.evaluation import (
    LONG_LABEL,
    SHORT_LABEL,
    make_origins,
    score_forecast,
    select_winner,
    split_at_origin,
    summarize,
)
from src.models import lstm_model
from src.models.base import BaseForecaster
from src.models.baseline import SeasonalNaiveForecaster
from src.models.gbdt import LightGBMForecaster
from src.models.lstm_model import LSTMForecaster
from src.models.prophet import ProphetForecaster
from src.models.statistical import HoltWintersForecaster

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Add a model by adding one line here. Each class is built with its default constructor args.
REGISTRY: dict[str, type[BaseForecaster]] = {
    "HoltWinters": HoltWintersForecaster,
    "Prophet": ProphetForecaster,
    "LightGBM": LightGBMForecaster,
    "LSTM": LSTMForecaster,
    BASELINE_NAME: SeasonalNaiveForecaster,
}

HORIZON_STEP_MINUTES = 30
METRICS = ["MAE", "RMSE", "MAPE", "sMAPE"]
ORIGIN_DATE_COLUMNS = ["origin", "train_start", "train_end", "test_start", "test_end"]
ORIGIN_COLUMNS = [
    "origin", "model", "horizon",
    "train_start", "train_end", "test_start", "test_end", "n_train",
    "fit_time_s", "predict_time_s",
    *METRICS, "error",
]
PREDICTION_COLUMNS = ["origin", "model", "timestamp", "step", "y_true", "y_pred"]

# Old single-holdout 30-day MAE at the last origin. The rolling run must reproduce these.
SANITY_ORIGIN = pd.Timestamp("2026-06-01 00:00")
SANITY_EXPECTED_MAE_30D = {"HoltWinters": 340.02, "Prophet": 179.86, "LightGBM": 205.35, "LSTM": 332.85}
SANITY_TOLERANCE = 0.01  # relative

SELECTION_TEXT = (
    "Models ranked by mean 7-day RMSE across rolling origins (baseline excluded). "
    "Paired Wilcoxon signed-rank test (alpha 0.05) on the top two: significant -> top model; "
    "otherwise the model with the lower mean fit time wins."
)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Running models
# ---------------------------------------------------------------------------

def run_one(name: str, origin: pd.Timestamp, y_train: pd.Series, y_test: pd.Series) -> tuple[list[dict], pd.DataFrame | None]:
    """
    Fit a fresh model on y_train, predict len(y_test) steps and score both horizons.

    Returns one result row per horizon and the long-format predictions (None on failure).
    A failure is recorded in the row's `error` column instead of stopping the run.
    """
    info = {
        "train_start": y_train.index[0],
        "train_end": y_train.index[-1],
        "test_start": y_test.index[0],
        "test_end": y_test.index[-1],
        "n_train": len(y_train),
        "fit_time_s": np.nan,
        "predict_time_s": np.nan,
    }
    scores = None
    forecast = None
    error = ""
    try:
        if y_test.isna().any():
            raise ValueError(f"test window has {int(y_test.isna().sum())} missing values")
        seed_everything()
        model = REGISTRY[name]()

        t0 = time.perf_counter()
        model.fit(y_train)
        info["fit_time_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        forecast = model.predict(horizon=len(y_test))
        info["predict_time_s"] = time.perf_counter() - t0

        if not forecast.index.equals(y_test.index):
            raise ValueError("prediction index does not match test index")
        if forecast.isna().any():
            raise ValueError(f"forecast has {int(forecast.isna().sum())} NaN values")
        scores = score_forecast(y_test.to_numpy(), forecast.to_numpy())
    except Exception as exc:  # noqa: BLE001 - keep benchmarking the other models
        error = f"{type(exc).__name__}: {exc}"
        forecast = None
        print(f"  ERROR in {name} (origin {origin:%Y-%m-%d}): {error}")
        traceback.print_exc()

    rows = []
    for label in (SHORT_LABEL, LONG_LABEL):
        row = {"origin": origin, "model": name, "horizon": label, **info, "error": error}
        for metric in METRICS:
            row[metric] = scores[label][metric] if scores is not None else np.nan
        rows.append(row)

    preds = None
    if forecast is not None:
        preds = pd.DataFrame({
            "origin": origin,
            "model": name,
            "timestamp": y_test.index,
            "step": np.arange(1, len(y_test) + 1),
            "y_true": y_test.to_numpy(),
            "y_pred": forecast.to_numpy(),
        })

    if scores is None:
        print(f"  {name:<13} origin {origin:%Y-%m-%d}: FAILED")
    else:
        print(
            f"  {name:<13} origin {origin:%Y-%m-%d}: "
            f"7d MAE={scores[SHORT_LABEL]['MAE']:.2f}  30d MAE={scores[LONG_LABEL]['MAE']:.2f}  "
            f"fit={info['fit_time_s']:.2f}s predict={info['predict_time_s']:.2f}s"
        )
    return rows, preds


# ---------------------------------------------------------------------------
# Output files (written after every origin so a crash loses nothing)
# ---------------------------------------------------------------------------

def _normalize(df: pd.DataFrame, date_columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])
    return df


def load_previous(origins_path: Path, predictions_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if origins_path.exists():
        results = pd.read_csv(origins_path)
        results["error"] = results["error"].fillna("")
    else:
        results = pd.DataFrame(columns=ORIGIN_COLUMNS)
    if predictions_path.exists():
        preds = pd.read_parquet(predictions_path)
    else:
        preds = pd.DataFrame(columns=PREDICTION_COLUMNS)
    return (
        _normalize(results, ORIGIN_DATE_COLUMNS),
        _normalize(preds, ["origin", "timestamp"]),
    )


def _atomic_write(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def write_outputs(results: pd.DataFrame, preds: pd.DataFrame, origins_path: Path, predictions_path: Path) -> None:
    _atomic_write(origins_path, lambda p: results.to_csv(p, index=False))
    _atomic_write(predictions_path, lambda p: preds.to_parquet(p, index=False))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def sanity_check(results: pd.DataFrame, names: list[str]) -> None:
    rows = results[(results["origin"] == SANITY_ORIGIN) & (results["horizon"] == LONG_LABEL)]
    if rows.empty:
        print(f"Sanity check skipped: origin {SANITY_ORIGIN:%Y-%m-%d} is not in this run.")
        return
    for name, expected in SANITY_EXPECTED_MAE_30D.items():
        if name not in names:
            print(f"Sanity check skipped for {name} (not in this run).")
            continue
        row = rows[rows["model"] == name]
        if row.empty or row["error"].iloc[0]:
            print(f"WARNING: sanity check could not run for {name} (no valid result at {SANITY_ORIGIN:%Y-%m-%d}).")
            continue
        actual = float(row["MAE"].iloc[0])
        rel_diff = abs(actual - expected) / expected
        if rel_diff > SANITY_TOLERANCE:
            print(
                f"WARNING: 30-day MAE for {name} at {SANITY_ORIGIN:%Y-%m-%d} is {actual:.2f}, "
                f"expected {expected:.2f} (off by {rel_diff:.2%}, tolerance {SANITY_TOLERANCE:.0%})."
            )
        else:
            print(f"Sanity check OK: {name} 30-day MAE at {SANITY_ORIGIN:%Y-%m-%d} = {actual:.2f} (expected {expected:.2f}).")


def print_table(summary: pd.DataFrame, label: str, title: str) -> None:
    s = summary[summary["horizon"] == label].sort_values("RMSE_mean")
    print(f"\n{title}: mean ± std across origins, ranked by mean RMSE")
    print("-" * 124)
    print(f"{'Model':<14}{'MAE':<24}{'RMSE':<24}{'MAPE %':<24}{'sMAPE %':<24}{'Fit s':<9}{'Wins':<6}{'Skill':<8}")
    print("-" * 124)

    def cell(row: pd.Series, col: str, digits: int) -> str:
        return f"{row[f'{col}_mean']:.{digits}f} ± {row[f'{col}_std']:.{digits}f}"

    for _, r in s.iterrows():
        print(
            f"{r['model']:<14}{cell(r, 'MAE', 2):<24}{cell(r, 'RMSE', 2):<24}"
            f"{cell(r, 'MAPE', 3):<24}{cell(r, 'sMAPE', 3):<24}"
            f"{r['fit_time_s_mean']:<9.2f}{int(r['wins']):<6}{r['skill_vs_baseline']:<8.3f}"
        )
    print("-" * 124)


def print_selection(selection: dict) -> None:
    print("\nSELECTION: mean 7-day RMSE across origins")
    for _, r in selection["ranking"].iterrows():
        print(f"  {int(r['rank'])}. {r['model']:<13} RMSE_7d={r['mean_rmse_7d']:.2f}  fit={r['mean_fit_time_s']:.2f}s")
    p = selection["p_value"]
    print(f"  Wilcoxon p (top two) = {'n/a' if p is None else f'{p:.4f}'}")
    print(f"  Winner: {selection['winner']} (runner-up: {selection['runner_up']}, rule: {selection['rule']})")


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

def feature_list(model: BaseForecaster) -> dict:
    """Lags and exogenous columns, read from the fitted model."""
    if isinstance(model, LightGBMForecaster):
        return {
            "lags": [int(lag) for lag in model.forecaster.lags],
            "exog": [str(c) for c in model.forecaster.exog_names_in_],
        }
    if isinstance(model, LSTMForecaster):
        return {
            "lags": list(range(1, model.lookback + 1)),
            "exog": list(lstm_model.CALENDAR_COLUMNS),
        }
    # Statistical, Prophet and seasonal-naive models use only the target history.
    return {"lags": [], "exog": []}


def metric_block(rows: pd.DataFrame, model: str, label: str) -> dict:
    """Mean and sample std across origins of each metric and of fit time."""
    g = rows[(rows["model"] == model) & (rows["horizon"] == label)]
    out = {}
    for col in METRICS + ["fit_time_s"]:
        values = g[col].astype(float)
        out[col] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else None,
        }
    return out


def wins_for(summary: pd.DataFrame, label: str, model: str) -> int:
    row = summary[(summary["horizon"] == label) & (summary["model"] == model)]
    return int(row["wins"].iloc[0])


def save_artifacts(
    winner: str,
    demand: pd.Series,
    rows: pd.DataFrame,
    origins: pd.DatetimeIndex,
    summary: pd.DataFrame,
    selection: dict,
    names: list[str],
) -> None:
    """Refit the winner on the full dataset and replace the artifacts only once everything succeeded."""
    cls = REGISTRY[winner]
    seed_everything()
    model = cls()
    model.fit(demand)

    meta = {
        "model_name": winner,
        "class_path": f"{cls.__module__}.{cls.__qualname__}",
        "selection": SELECTION_TEXT,
        "backtest_336_metrics": metric_block(rows, winner, SHORT_LABEL),
        "holdout_1440_metrics": metric_block(rows, winner, LONG_LABEL),
        "training_date_range": {
            "start": str(demand.index[0]),
            "end": str(demand.index[-1]),
            "n_rows": len(demand),
        },
        "horizon_step_minutes": HORIZON_STEP_MINUTES,
        "seed": SEED,
        "features": feature_list(model),
        "artifact_path": str(BEST_MODEL_PATH.relative_to(ROOT)),
        "n_origins": len(origins),
        "wins_7d": wins_for(summary, SHORT_LABEL, winner),
        "wins_30d": wins_for(summary, LONG_LABEL, winner),
        "wilcoxon_p": selection["p_value"],
        "selection_rule": selection["rule"],
        "lstm_included": "LSTM" in names,
        "evaluation": (
            f"Rolling-origin backtest: {len(origins)} monthly origins from "
            f"{origins[0]:%Y-%m-%d} to {origins[-1]:%Y-%m-%d} (00:00). Each model is refit on all data "
            f"up to the origin and forecasts {LONG_HORIZON} steps (30 days); 7-day metrics use the first "
            f"{SHORT_HORIZON} steps. Metrics are mean and sample std across origins."
        ),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_model = BEST_MODEL_PATH.with_name(BEST_MODEL_PATH.name + ".tmp")
    tmp_meta = BEST_MODEL_META_PATH.with_name(BEST_MODEL_META_PATH.name + ".tmp")
    try:
        model.save(str(tmp_model))
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
        os.replace(tmp_model, BEST_MODEL_PATH)
        os.replace(tmp_meta, BEST_MODEL_META_PATH)
    except BaseException:
        for p in (tmp_model, tmp_meta):
            p.unlink(missing_ok=True)
        raise
    print(f"Saved {BEST_MODEL_PATH.relative_to(ROOT)} and {BEST_MODEL_META_PATH.relative_to(ROOT)} ({winner}).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-lstm", action="store_true", help="Do not run the LSTM model.")
    parser.add_argument("--n-origins", type=int, default=N_ORIGINS,
                        help=f"Use only the last N origins (1-{N_ORIGINS}, default {N_ORIGINS}).")
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR,
                        help="Directory for benchmark_origins.csv, predictions_origins.parquet and benchmark_results.csv.")
    parser.add_argument("--no-artifacts", action="store_true",
                        help="Skip refitting the winner and saving it to artifacts/.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip origin and model pairs already present in benchmark_origins.csv.")
    args = parser.parse_args()
    if not 1 <= args.n_origins <= N_ORIGINS:
        parser.error(f"--n-origins must be between 1 and {N_ORIGINS}")
    return args


def main() -> None:
    args = parse_args()
    names = [n for n in REGISTRY if not (args.skip_lstm and n == "LSTM")]
    origins = make_origins()[-args.n_origins:]

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    origins_path = out_dir / BENCHMARK_ORIGINS_CSV.name
    predictions_path = out_dir / PREDICTIONS_PARQUET.name
    results_path = out_dir / BENCHMARK_RESULTS_CSV.name

    demand = load_processed_dataset(DEMAND_PARQUET_PATH)["Demand"]
    print(f"Data: {len(demand)} rows, {demand.index[0]} to {demand.index[-1]}")
    print(f"Origins: {len(origins)} monthly, {origins[0]:%Y-%m-%d} to {origins[-1]:%Y-%m-%d}")
    print(f"Models: {', '.join(names)} | seed={SEED} | horizons: {SHORT_HORIZON} and {LONG_HORIZON} steps")
    print(f"Output dir: {out_dir}")

    if args.resume:
        results, preds = load_previous(origins_path, predictions_path)
        print(f"Resume: {len(results)} rows already in {origins_path.name}")
    else:
        results = pd.DataFrame(columns=ORIGIN_COLUMNS)
        preds = pd.DataFrame(columns=PREDICTION_COLUMNS)

    for origin in origins:
        done = set(zip(results["origin"], results["model"]))
        todo = [n for n in names if (origin, n) not in done]
        skipped = [n for n in names if (origin, n) in done]
        if skipped:
            print(f"\nResume: skipping {', '.join(skipped)} at origin {origin:%Y-%m-%d}")
        if not todo:
            continue

        y_train, y_test = split_at_origin(demand, origin)
        print(
            f"\nORIGIN {origin:%Y-%m-%d}: train {y_train.index[0]} -> {y_train.index[-1]} ({len(y_train)} steps), "
            f"test {y_test.index[0]} -> {y_test.index[-1]} ({len(y_test)} steps)"
        )
        new_rows: list[dict] = []
        new_preds: list[pd.DataFrame] = []
        for name in todo:
            rows, pred_df = run_one(name, origin, y_train, y_test)
            new_rows.extend(rows)
            if pred_df is not None:
                new_preds.append(pred_df)

        results = _normalize(
            pd.concat([results, pd.DataFrame(new_rows, columns=ORIGIN_COLUMNS)], ignore_index=True),
            ORIGIN_DATE_COLUMNS,
        )
        preds = _normalize(pd.concat([preds, *new_preds], ignore_index=True), ["origin", "timestamp"])
        write_outputs(results, preds, origins_path, predictions_path)
        print(f"Wrote {origins_path.name} ({len(results)} rows) and {predictions_path.name} ({len(preds)} rows)")

    current = results[results["origin"].isin(origins)].copy()

    failed = current[current["error"] != ""]
    if (failed["model"] == BASELINE_NAME).any():
        raise RuntimeError(f"{BASELINE_NAME} failed at some origins; the benchmark cannot be scored.")
    excluded = sorted(set(failed["model"]))
    for name in excluded:
        n_bad = failed.loc[failed["model"] == name, "origin"].nunique()
        print(f"WARNING: {name} failed at {n_bad} origin(s); excluded from the summary and selection.")
    ok = current[~current["model"].isin(excluded)]

    summary = summarize(ok, BASELINE_NAME)
    summary.to_csv(results_path, index=False)
    selection = select_winner(ok, BASELINE_NAME)
    winner = selection["winner"]

    print()
    sanity_check(current, names)
    print_table(summary, SHORT_LABEL, f"7-DAY ({SHORT_HORIZON} steps)")
    print_table(summary, LONG_LABEL, f"30-DAY ({LONG_HORIZON} steps)")
    print_selection(selection)

    if args.no_artifacts:
        print("\nSkipped refit and artifacts (--no-artifacts).")
    else:
        print()
        save_artifacts(winner, demand, ok, origins, summary, selection, names)

    print(f"\nWrote {origins_path}, {predictions_path}, {results_path}")


if __name__ == "__main__":
    main()
