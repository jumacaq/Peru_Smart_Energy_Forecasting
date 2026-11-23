"""
Rolling-origin evaluation for the monthly benchmark.

Each origin splits the series into an expanding training window (up to and
including the origin) and a 30-day test window that starts 30 minutes later.
Per-origin scores are then summarized across origins and a winner is chosen
among the non-baseline models.
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.config import (
    BASELINE_NAME,
    FIRST_ORIGIN,
    LONG_HORIZON,
    N_ORIGINS,
    SHORT_HORIZON,
)
from src.metrics import evaluate_all

SHORT_LABEL = "7d"
LONG_LABEL = "30d"
STEP = pd.Timedelta(minutes=30)
METRIC_COLUMNS = ["MAE", "RMSE", "MAPE", "sMAPE"]
RESULT_COLUMNS = ["origin", "model", "horizon", *METRIC_COLUMNS, "fit_time_s"]


def make_origins(
    first: pd.Timestamp = FIRST_ORIGIN, n: int = N_ORIGINS
) -> pd.DatetimeIndex:
    """
    Return monthly forecast origins at 00:00 on the first day of each month.

    Parameters
    ----------
    first : pd.Timestamp
        First origin (default: FIRST_ORIGIN from config).
    n : int
        Number of origins (default: N_ORIGINS from config).

    Returns
    -------
    pd.DatetimeIndex
        Origins from `first` spaced one calendar month apart.
    """
    return pd.date_range(start=first, periods=n, freq="MS")


def split_at_origin(y: pd.Series, origin) -> tuple[pd.Series, pd.Series]:
    """
    Split a series into an expanding training window and a 30-day test window.

    Parameters
    ----------
    y : pd.Series
        Target values with a sorted DatetimeIndex at 30-minute steps.
    origin : pd.Timestamp or str
        Forecast origin. It is included in the training window.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (train, test). Train ends at `origin`; test starts at origin + 30 min
        and has LONG_HORIZON steps.

    Raises
    ------
    TypeError
        If y lacks a DatetimeIndex.
    ValueError
        If there is no data up to the origin or the test window is incomplete.
    """
    if not isinstance(y.index, pd.DatetimeIndex):
        raise TypeError("y must have a DatetimeIndex")

    origin = pd.Timestamp(origin)
    train = y.loc[:origin]
    if train.empty:
        raise ValueError(f"No training data at or before origin {origin}")

    test_index = pd.date_range(origin + STEP, periods=LONG_HORIZON, freq="30min")
    missing = test_index.difference(y.index)
    if len(missing) > 0:
        raise ValueError(
            f"Test window after origin {origin} is incomplete: "
            f"{len(missing)} of {LONG_HORIZON} steps missing "
            f"(first missing: {missing[0]})"
        )

    test = y.reindex(test_index)
    return train, test


def score_forecast(y_true, y_pred) -> dict[str, dict[str, float]]:
    """
    Score a forecast over the 7-day and 30-day horizons.

    Values are matched by position, so y_true[i] is compared with y_pred[i].

    Parameters
    ----------
    y_true : array-like
        Actual values for the test window (LONG_HORIZON steps).
    y_pred : array-like
        Forecast values for the same steps.

    Returns
    -------
    dict
        {"7d": metrics over the first SHORT_HORIZON steps,
         "30d": metrics over all LONG_HORIZON steps}, each with
        MAE, RMSE, MAPE and sMAPE.

    Raises
    ------
    ValueError
        If the inputs differ in shape or do not have LONG_HORIZON steps.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
    if len(yt) != LONG_HORIZON:
        raise ValueError(f"Expected {LONG_HORIZON} steps, got {len(yt)}")

    return {
        SHORT_LABEL: evaluate_all(yt[:SHORT_HORIZON], yp[:SHORT_HORIZON]),
        LONG_LABEL: evaluate_all(yt, yp),
    }


def _check_results(origin_results: pd.DataFrame) -> None:
    missing_cols = set(RESULT_COLUMNS) - set(origin_results.columns)
    if missing_cols:
        raise ValueError(f"origin_results is missing columns: {sorted(missing_cols)}")
    if origin_results.duplicated(["origin", "model", "horizon"]).any():
        raise ValueError("origin_results has duplicate (origin, model, horizon) rows")


def summarize(origin_results: pd.DataFrame, baseline: str = BASELINE_NAME) -> pd.DataFrame:
    """
    Summarize per-origin results by model and horizon.

    Parameters
    ----------
    origin_results : pd.DataFrame
        Long table with columns origin, model, horizon, MAE, RMSE, MAPE, sMAPE,
        fit_time_s. Must include the baseline model for every origin and horizon.
    baseline : str
        Name of the baseline model.

    Returns
    -------
    pd.DataFrame
        One row per (model, horizon) with columns:
        - <metric>_mean, <metric>_std, <metric>_median for each metric
        - fit_time_s_mean
        - wins: origins where the model had the lowest RMSE among non-baseline models
        - skill_vs_baseline: mean over origins of 1 - RMSE / RMSE_baseline

    Raises
    ------
    ValueError
        If columns are missing, rows are duplicated, or the baseline is absent
        for some origin and horizon.
    """
    _check_results(origin_results)
    if baseline not in set(origin_results["model"]):
        raise ValueError(f"Baseline '{baseline}' is missing from origin_results")

    df = origin_results.merge(
        origin_results.loc[origin_results["model"] == baseline, ["origin", "horizon", "RMSE"]]
        .rename(columns={"RMSE": "baseline_RMSE"}),
        on=["origin", "horizon"],
        how="left",
        validate="many_to_one",
    )
    if df["baseline_RMSE"].isna().any():
        raise ValueError("Every origin and horizon needs a baseline row")
    df["skill"] = 1 - df["RMSE"] / df["baseline_RMSE"]

    real = df[df["model"] != baseline]
    if real.empty:
        winners = pd.DataFrame(columns=["horizon", "model"])
    else:
        best_rows = real.groupby(["horizon", "origin"])["RMSE"].idxmin()
        winners = real.loc[best_rows, ["horizon", "model"]]

    grouped = df.groupby(["horizon", "model"])
    summary = grouped[METRIC_COLUMNS].agg(["mean", "std", "median"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["fit_time_s_mean"] = grouped["fit_time_s"].mean()
    summary["skill_vs_baseline"] = grouped["skill"].mean()
    wins = winners.groupby(["horizon", "model"]).size()
    summary["wins"] = wins.reindex(summary.index, fill_value=0).astype(int)

    return summary.reset_index().sort_values(["horizon", "model"]).reset_index(drop=True)


def select_winner(
    origin_results: pd.DataFrame,
    baseline: str = BASELINE_NAME,
    alpha: float = 0.05,
) -> dict:
    """
    Choose the winning non-baseline model from 7-day RMSE across origins.

    Models are ranked by mean 7-day RMSE. A paired Wilcoxon signed-rank test on
    the per-origin 7-day RMSEs of the top two decides between them:
    - p < alpha: the top-ranked model wins (rule "wilcoxon_significant").
    - p >= alpha: the difference is treated as a tie and the model with the lower
      mean fit time wins (rule "wilcoxon_tie_fit_time").
    With a single non-baseline model, that model wins (rule "single_model").

    Parameters
    ----------
    origin_results : pd.DataFrame
        Long table as described in `summarize`.
    baseline : str
        Model excluded from the ranking.
    alpha : float
        Significance level for the Wilcoxon test.

    Returns
    -------
    dict
        Keys: winner, runner_up (None if single model), p_value (None if single
        model), rule, ranking (DataFrame with rank, model, mean_rmse_7d,
        mean_fit_time_s).

    Raises
    ------
    ValueError
        If there are no non-baseline models with 7-day results, or the top two
        models do not share the same origins.
    """
    _check_results(origin_results)
    short = origin_results[
        (origin_results["horizon"] == SHORT_LABEL) & (origin_results["model"] != baseline)
    ]
    if short.empty:
        raise ValueError(f"No non-baseline models with {SHORT_LABEL} results")

    ranking = (
        short.groupby("model")
        .agg(mean_rmse_7d=("RMSE", "mean"), mean_fit_time_s=("fit_time_s", "mean"))
        .reset_index()
        .sort_values(["mean_rmse_7d", "model"])
        .reset_index(drop=True)
    )
    ranking.insert(0, "rank", range(1, len(ranking) + 1))

    models = ranking["model"].tolist()
    if len(models) == 1:
        return {
            "winner": models[0],
            "runner_up": None,
            "p_value": None,
            "rule": "single_model",
            "ranking": ranking,
        }

    top, second = models[0], models[1]
    pivot = short.pivot(index="origin", columns="model", values="RMSE")[[top, second]]
    if pivot.isna().any().any():
        raise ValueError("The top two models must share the same origins")

    rmse_top = pivot[top].to_numpy()
    rmse_second = pivot[second].to_numpy()
    if np.all(rmse_top == rmse_second):
        p_value = 1.0
    else:
        p_value = float(wilcoxon(rmse_top, rmse_second).pvalue)

    if p_value < alpha:
        winner, rule = top, "wilcoxon_significant"
    else:
        fit_time = ranking.set_index("model")["mean_fit_time_s"]
        winner = top if fit_time[top] <= fit_time[second] else second
        rule = "wilcoxon_tie_fit_time"
    runner_up = second if winner == top else top

    return {
        "winner": winner,
        "runner_up": runner_up,
        "p_value": p_value,
        "rule": rule,
        "ranking": ranking,
    }
