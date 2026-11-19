"""
Benchmark all registered forecasters and save the best one.

Protocol
--------
* Backtest: four consecutive, non-overlapping 336-step (7-day) windows that end at the
  end of the data. For each window every model is refit from scratch on all data
  before the window (expanding window) and predicts the 336 steps of the window.
* Holdout: one 1,440-step (30-day) window at the end of the data, trained on everything
  before it.
* Ranking: mean RMSE over the four 336-step windows (holdout is reported, not ranked).
* Artifacts: the top-ranked model is refit on the full dataset and saved via its own
  save() to artifacts/best_forecaster.pkl, with metadata in artifacts/best_forecaster_meta.json.

Usage
-----
    python scripts/run_benchmark.py              # all models
    python scripts/run_benchmark.py --skip-lstm  # skip the (slow) LSTM
"""

import argparse
import json
import logging
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
    BENCHMARK_RESULTS_CSV,
    BENCHMARK_WINDOWS_CSV,
    BEST_MODEL_META_PATH,
    BEST_MODEL_PATH,
    DEMAND_PARQUET_PATH,
    REPORTS_DIR,
    ROOT,
)
from src.data_loader import load_processed_dataset  # noqa: E402
from src.metrics import evaluate_all  # noqa: E402
from src.models import lstm_model  # noqa: E402
from src.models.base import BaseForecaster  # noqa: E402
from src.models.gbdt import LightGBMForecaster  # noqa: E402
from src.models.lstm_model import LSTMForecaster  # noqa: E402
from src.models.prophet import ProphetForecaster  # noqa: E402
from src.models.statistical import HoltWintersForecaster  # noqa: E402

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
}

SEED = 42
HORIZON_STEP_MINUTES = 30
BACKTEST_HORIZON = 336
N_BACKTEST_WINDOWS = 4
HOLDOUT_HORIZON = 1440
METRICS = ["MAE", "RMSE", "MAPE", "sMAPE"]

# Reference values from the earlier single-window benchmark (docs/benchmark_results.csv,
# 7-day rows). The last 336-step backtest window must reproduce these.
SANITY_EXPECTED_MAE = {"LightGBM": 116.58, "Prophet": 218.10}
SANITY_TOLERANCE = 0.01


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Running models
# ---------------------------------------------------------------------------

def run_one(name: str, y_train: pd.Series, y_test: pd.Series, split: str, window_id: int) -> dict:
    """Fit a fresh model on y_train, predict len(y_test) steps, score against y_test."""
    row = {
        "split": split,
        "window_id": window_id,
        "model": name,
        "train_start": y_train.index[0],
        "train_end": y_train.index[-1],
        "test_start": y_test.index[0],
        "test_end": y_test.index[-1],
        "horizon": len(y_test),
    }
    try:
        seed_everything()
        model = REGISTRY[name]()
        t0 = time.perf_counter()
        model.fit(y_train)
        row["fit_time_s"] = time.perf_counter() - t0

        pred = model.predict(horizon=len(y_test))
        if not pred.index.equals(y_test.index):
            raise ValueError("prediction index does not match test index")
        row.update(evaluate_all(y_test, pred))
        row["error"] = ""
    except Exception as exc:  # keep benchmarking the other models
        row["error"] = f"{type(exc).__name__}: {exc}"
        print(f"  ERROR in {name} ({split}, window {window_id}): {row['error']}")
        traceback.print_exc()
    print(
        f"  {name:<11} {split:<14} window {window_id}: "
        + (
            f"MAE={row['MAE']:.4f} RMSE={row['RMSE']:.4f} fit={row['fit_time_s']:.2f}s"
            if not row["error"]
            else "FAILED"
        )
    )
    return row


def summarize(raw: pd.DataFrame, split: str) -> pd.DataFrame:
    """
    Mean and sample std per model over the windows of one split.
    Models with any failed window are listed but get no rank.
    """
    split_df = raw[raw["split"] == split]
    n_expected = split_df["window_id"].nunique()
    ok = split_df[split_df["error"] == ""]

    rows = []
    for name in REGISTRY:
        if name not in set(split_df["model"]):
            continue
        g = ok[ok["model"] == name]
        entry = {"split": split, "model": name, "n_windows": len(g), "n_expected": n_expected}
        for col in METRICS + ["fit_time_s"]:
            entry[f"{col}_mean"] = g[col].mean() if len(g) else np.nan
            entry[f"{col}_std"] = g[col].std(ddof=1) if len(g) > 1 else np.nan
        rows.append(entry)

    summary = pd.DataFrame(rows)
    summary["rank"] = np.nan
    if split == "backtest":  # the holdout is reported only; ranking uses the backtest
        complete = summary["n_windows"] == n_expected
        ranked = summary[complete].sort_values("RMSE_mean").index
        summary.loc[ranked, "rank"] = range(1, len(ranked) + 1)
    summary["rank"] = summary["rank"].astype("Int64")
    return summary.sort_values(["rank", "model"], na_position="last").reset_index(drop=True)


def sanity_check(raw: pd.DataFrame) -> None:
    last = raw[(raw["split"] == "backtest") & (raw["window_id"] == N_BACKTEST_WINDOWS)]
    for name, expected in SANITY_EXPECTED_MAE.items():
        row = last[last["model"] == name]
        if row.empty or row["error"].iloc[0]:
            print(f"WARNING: sanity check could not run for {name} (no valid last-window result).")
            continue
        actual = float(row["MAE"].iloc[0])
        if abs(actual - expected) > SANITY_TOLERANCE:
            print(
                f"WARNING: last 336-step window MAE for {name} is {actual:.4f}, "
                f"expected about {expected:.2f}. Earlier results do not reproduce."
            )
        else:
            print(f"Sanity check OK: {name} last-window MAE = {actual:.4f} (expected ~{expected:.2f}).")


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
    # HoltWinters and Prophet use only the target history; no exogenous inputs.
    return {"lags": [], "exog": []}


def save_best(ranking: pd.Series, full: pd.Series, summary_bt: pd.Series, summary_ho: pd.Series) -> None:
    name = ranking["model"]
    cls = REGISTRY[name]

    seed_everything()
    model = cls()
    model.fit(full)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(BEST_MODEL_PATH))

    def metric_block(s: pd.Series) -> dict:
        out = {}
        for col in METRICS + ["fit_time_s"]:
            out[col] = {"mean": float(s[f"{col}_mean"]), "std": float(s[f"{col}_std"]) if pd.notna(s[f"{col}_std"]) else None}
        return out

    meta = {
        "model_name": name,
        "class_path": f"{cls.__module__}.{cls.__qualname__}",
        "selection": "lowest mean RMSE over the four 336-step backtest windows",
        "backtest_336_metrics": metric_block(summary_bt),
        "holdout_1440_metrics": metric_block(summary_ho),
        "training_date_range": {
            "start": str(full.index[0]),
            "end": str(full.index[-1]),
            "n_rows": int(len(full)),
        },
        "horizon_step_minutes": HORIZON_STEP_MINUTES,
        "seed": SEED,
        "features": feature_list(model),
        "artifact_path": str(BEST_MODEL_PATH.relative_to(ROOT)),
    }
    with open(BEST_MODEL_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"Saved {BEST_MODEL_PATH.relative_to(ROOT)} and {BEST_MODEL_META_PATH.relative_to(ROOT)} ({name}).")


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_table(summary: pd.DataFrame, title: str) -> None:
    print(f"\n{title}")
    print("-" * 118)
    print(f"{'Rank':<6}{'Model':<12}{'MAE':<22}{'RMSE':<22}{'MAPE %':<20}{'sMAPE %':<20}{'Fit time s':<18}{'Windows':<8}")
    print("-" * 118)
    for _, r in summary.iterrows():
        def cell(col: str, digits: int) -> str:
            mean = r[f"{col}_mean"]
            std = r[f"{col}_std"]
            if pd.isna(mean):
                return "ERROR"
            if pd.isna(std):
                return f"{mean:.{digits}f}"
            return f"{mean:.{digits}f} ± {std:.{digits}f}"

        rank = "-" if pd.isna(r["rank"]) else str(int(r["rank"]))
        print(
            f"{rank:<6}{r['model']:<12}{cell('MAE', 2):<22}{cell('RMSE', 2):<22}"
            f"{cell('MAPE', 3):<20}{cell('sMAPE', 3):<20}{cell('fit_time_s', 2):<18}"
            f"{r['n_windows']}/{r['n_expected']}"
        )
    print("-" * 118)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global REGISTRY
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-lstm", action="store_true", help="Do not run the LSTM model.")
    args = parser.parse_args()

    if args.skip_lstm:
        REGISTRY = {n: c for n, c in REGISTRY.items() if n != "LSTM"}
    names = list(REGISTRY)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    demand = load_processed_dataset(DEMAND_PARQUET_PATH)["Demand"]
    n = len(demand)
    print(f"Data: {n} rows, {demand.index[0]} to {demand.index[-1]}")
    print(f"Models: {', '.join(names)} | seed={SEED}")

    raw_rows: list[dict] = []

    print(f"\nBACKTEST: {N_BACKTEST_WINDOWS} x {BACKTEST_HORIZON}-step windows (expanding window)")
    for k in range(1, N_BACKTEST_WINDOWS + 1):
        end = n - (N_BACKTEST_WINDOWS - k) * BACKTEST_HORIZON
        start = end - BACKTEST_HORIZON
        y_train = demand.iloc[:start]
        y_test = demand.iloc[start:end]
        print(f"Window {k}: train {y_train.index[0]} -> {y_train.index[-1]}, "
              f"test {y_test.index[0]} -> {y_test.index[-1]}")
        for name in REGISTRY:
            raw_rows.append(run_one(name, y_train, y_test, "backtest", k))
        # Save raw results as each window completes.
        pd.DataFrame(raw_rows).to_csv(BENCHMARK_WINDOWS_CSV, index=False)

    print(f"\nHOLDOUT: {HOLDOUT_HORIZON}-step window")
    y_train = demand.iloc[:-HOLDOUT_HORIZON]
    y_test = demand.iloc[-HOLDOUT_HORIZON:]
    print(f"train {y_train.index[0]} -> {y_train.index[-1]}, test {y_test.index[0]} -> {y_test.index[-1]}")
    for name in REGISTRY:
        raw_rows.append(run_one(name, y_train, y_test, "holdout", 1))
    raw = pd.DataFrame(raw_rows)
    raw.to_csv(BENCHMARK_WINDOWS_CSV, index=False)

    sanity_check(raw)

    summary_bt = summarize(raw, "backtest")
    summary_ho = summarize(raw, "holdout")
    summary = pd.concat([summary_bt, summary_ho], ignore_index=True)
    summary.to_csv(BENCHMARK_RESULTS_CSV, index=False)

    ranked_bt = summary_bt[summary_bt["rank"].notna()].sort_values("rank")
    if ranked_bt.empty:
        raise RuntimeError("No model completed all backtest windows; cannot select a best model.")
    best = ranked_bt.iloc[0]
    print(f"\nBest model by mean backtest RMSE: {best['model']}")

    save_best(
        best,
        demand,
        summary_bt[summary_bt["model"] == best["model"]].iloc[0],
        summary_ho[summary_ho["model"] == best["model"]].iloc[0],
    )

    print_table(summary_bt, f"BACKTEST: {N_BACKTEST_WINDOWS} x {BACKTEST_HORIZON} steps, mean ± std over windows (ranked by mean RMSE)")
    print_table(summary_ho, f"HOLDOUT: {HOLDOUT_HORIZON} steps (single window, reported only; ranking uses the backtest)")
    print(f"\nWrote {BENCHMARK_WINDOWS_CSV.relative_to(ROOT)}, {BENCHMARK_RESULTS_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
