"""
Model comparison for the rolling-origin benchmark (pure functions, no Streamlit).

Means, stds, wins, skill and the top-two Wilcoxon come from src.evaluation
(summarize and select_winner). This module adds the per-model paired Wilcoxon test
against the baseline, the worst-origin RMSE, the verdict structure and its
plain-English text, and table shapes for the app.

Models that failed, or are missing, at any origin are excluded from the comparison
and listed in the verdict, as scripts/run_benchmark.py does.
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.config import BASELINE_NAME, LONG_HORIZON, SHORT_HORIZON
from src.evaluation import (
    LONG_LABEL,
    METRIC_COLUMNS,
    RESULT_COLUMNS,
    SHORT_LABEL,
    select_winner,
    summarize,
)

HORIZONS = (SHORT_LABEL, LONG_LABEL)
HORIZON_DAYS = {SHORT_LABEL: "7 days", LONG_LABEL: "30 days"}
HORIZON_STEPS = {SHORT_LABEL: SHORT_HORIZON, LONG_LABEL: LONG_HORIZON}
ALPHA = 0.05


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def _is_ok(df: pd.DataFrame) -> pd.Series:
    """True for rows with no error message and finite metrics."""
    ok = df[METRIC_COLUMNS].notna().all(axis=1)
    if "error" in df.columns:
        ok &= df["error"].fillna("").astype(str).str.strip() == ""
    return ok


def clean_results(
    origin_results: pd.DataFrame, baseline: str = BASELINE_NAME
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Drop failed rows and exclude every model that is not complete at all origins.

    Parameters
    ----------
    origin_results : pd.DataFrame
        Long table with the columns of src.evaluation.RESULT_COLUMNS, plus an optional
        `error` column (empty string or NaN when the run succeeded).
    baseline : str
        Baseline model. It must be complete, because every comparison is against it.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, str]]
        (valid rows of the complete models, {excluded model: reason}).

    Raises
    ------
    ValueError
        If columns are missing, or the baseline is absent or incomplete.
    """
    missing = set(RESULT_COLUMNS) - set(origin_results.columns)
    if missing:
        raise ValueError(f"origin_results is missing columns: {sorted(missing)}")

    df = origin_results.copy()
    ok = _is_ok(df)
    n_origins = df["origin"].nunique()

    good_counts = df[ok].groupby(["model", "horizon"])["origin"].nunique()
    excluded: dict[str, str] = {}
    for model in sorted(df["model"].unique()):
        worst_missing = max(
            n_origins - int(good_counts.get((model, horizon), 0))
            for horizon in HORIZONS
        )
        if worst_missing > 0:
            excluded[model] = (
                f"failed or missing at {worst_missing} of {n_origins} origins"
            )

    if baseline not in set(df["model"]):
        raise ValueError(f"Baseline '{baseline}' is missing from the results")
    if baseline in excluded:
        raise ValueError(
            f"Baseline '{baseline}' {excluded[baseline]}, so the comparison cannot be scored"
        )

    valid = df[ok & ~df["model"].isin(excluded)].copy()
    return valid, excluded


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _paired_p(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided paired Wilcoxon p-value. Identical values give 1.0, as in select_winner."""
    if np.all(a == b):
        return 1.0
    return float(wilcoxon(a, b).pvalue)


def _horizon_verdict(
    valid: pd.DataFrame,
    summary: pd.DataFrame,
    horizon: str,
    baseline: str,
    alpha: float,
    n_origins: int,
    excluded: dict[str, str],
) -> dict:
    h_valid = valid[valid["horizon"] == horizon]
    stats = summary[summary["horizon"] == horizon].set_index("model")
    worst = h_valid.groupby("model")["RMSE"].max()
    pivot = h_valid.pivot(index="origin", columns="model", values="RMSE").sort_index()
    base_rmse = pivot[baseline].to_numpy()

    # select_winner always scores the 7-day rows, so pass this horizon's rows relabelled.
    selection = select_winner(h_valid.assign(horizon=SHORT_LABEL), baseline, alpha)
    candidates = selection["ranking"]["model"].tolist()

    rows = []
    for rank, model in enumerate(candidates, start=1):
        p = _paired_p(pivot[model].to_numpy(), base_rmse)
        s = stats.loc[model]
        rows.append(
            {
                "rank": rank,
                "model": model,
                "is_baseline": False,
                "MAE_mean": s["MAE_mean"],
                "RMSE_mean": s["RMSE_mean"],
                "RMSE_std": s["RMSE_std"],
                "RMSE_worst": worst[model],
                "wins": int(s["wins"]),
                "skill_vs_baseline": s["skill_vs_baseline"],
                "fit_time_s_mean": s["fit_time_s_mean"],
                "p_vs_baseline": p,
                # "Beats" needs a lower mean RMSE as well as a significant two-sided test.
                "significant_vs_baseline": bool(
                    p < alpha and s["RMSE_mean"] < stats.loc[baseline, "RMSE_mean"]
                ),
            }
        )
    b = stats.loc[baseline]
    rows.append(
        {
            "rank": None,
            "model": baseline,
            "is_baseline": True,
            "MAE_mean": b["MAE_mean"],
            "RMSE_mean": b["RMSE_mean"],
            "RMSE_std": b["RMSE_std"],
            "RMSE_worst": worst[baseline],
            "wins": None,
            "skill_vs_baseline": None,
            "fit_time_s_mean": b["fit_time_s_mean"],
            "p_vs_baseline": None,
            "significant_vs_baseline": False,
        }
    )
    ranking = pd.DataFrame(rows)
    ranking["rank"] = ranking["rank"].astype("Int64")

    # The top-two p-value compares ranks 1 and 2 by mean RMSE. select_winner's own
    # runner_up is not used: after a fit-time tie it can be the rank-1 model.
    top_p = selection["p_value"]
    return {
        "horizon": horizon,
        "baseline": baseline,
        "alpha": alpha,
        "n_origins": n_origins,
        "excluded": dict(excluded),
        "ranking": ranking,
        "best_model": candidates[0],
        "runner_up": candidates[1] if len(candidates) > 1 else None,
        "top_two_p_value": top_p,
        "clear_winner": top_p is not None and top_p < alpha,
        "significant_vs_baseline": ranking.loc[
            ranking["significant_vs_baseline"], "model"
        ].tolist(),
    }


def build_verdict(
    origin_results: pd.DataFrame,
    baseline: str = BASELINE_NAME,
    alpha: float = ALPHA,
) -> dict[str, dict]:
    """
    Compare the models over all origins, separately for the 7-day and 30-day horizons.

    Parameters
    ----------
    origin_results : pd.DataFrame
        Long table as described in `clean_results`.
    baseline : str
        Model used as the reference. It is not ranked.
    alpha : float
        Significance level for every Wilcoxon test.

    Returns
    -------
    dict[str, dict]
        {"7d": verdict, "30d": verdict}. Each verdict has:
        - ranking: DataFrame, one row per model. Non-baseline models come first in
          mean-RMSE order (rank 1 is best), then the baseline. Columns: rank, model,
          is_baseline, MAE_mean, RMSE_mean, RMSE_std, RMSE_worst, wins,
          skill_vs_baseline, fit_time_s_mean, p_vs_baseline (paired Wilcoxon of
          per-origin RMSE against the baseline), significant_vs_baseline.
        - best_model, runner_up: top two non-baseline models by mean RMSE.
        - top_two_p_value: paired Wilcoxon p of the top two (None with one model).
        - clear_winner: True when top_two_p_value < alpha.
        - significant_vs_baseline: models with a lower mean RMSE than the baseline
          and a p-value below alpha.
        - excluded: {model: reason} for models left out, and n_origins, baseline, alpha.

    Raises
    ------
    ValueError
        If no non-baseline model is complete, or the baseline is not.
    """
    valid, excluded = clean_results(origin_results, baseline)
    if valid.loc[valid["model"] != baseline, "model"].nunique() == 0:
        raise ValueError("No non-baseline model has complete results")

    summary = summarize(valid, baseline)
    n_origins = valid["origin"].nunique()
    return {
        horizon: _horizon_verdict(
            valid, summary, horizon, baseline, alpha, n_origins, excluded
        )
        for horizon in HORIZONS
    }


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


def _join(names: list[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _p_phrase(p: float) -> str:
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def verdict_text(verdict: dict[str, dict]) -> list[str]:
    """
    Plain-English sentences describing a build_verdict result.

    Each horizon gives a ranking sentence and a baseline sentence. A final sentence
    lists any excluded models. No model names or numbers are hardcoded.
    """
    sentences: list[str] = []
    excluded: dict[str, str] = {}
    for horizon in HORIZONS:
        v = verdict[horizon]
        days = HORIZON_DAYS[horizon]
        alpha = v["alpha"]
        baseline = v["baseline"]
        excluded = v["excluded"]
        ranking = v["ranking"]
        best = ranking.iloc[0]

        if v["runner_up"] is None:
            sentences.append(
                f"{v['best_model']} is the only model with complete results at {days}, "
                f"so there is no pairwise comparison."
            )
        else:
            head = (
                f"{v['best_model']} has the lowest mean RMSE at {days} "
                f"({best['RMSE_mean']:.2f} ± {best['RMSE_std']:.2f})"
            )
            p_text = _p_phrase(v["top_two_p_value"])
            if v["clear_winner"]:
                sentences.append(
                    f"{head}, and its difference from {v['runner_up']} is statistically "
                    f"significant ({p_text})."
                )
            else:
                sentences.append(
                    f"{head}, but its difference from {v['runner_up']} is not statistically "
                    f"significant ({p_text}), so there is no clear winner."
                )

        sig = v["significant_vs_baseline"]
        if not sig:
            sentences.append(
                f"No model significantly beats the {baseline} baseline at {days}."
            )
        else:
            verb = "beats" if len(sig) == 1 else "beat"
            sentences.append(
                f"{_join(sig)} significantly {verb} the {baseline} baseline at {days} "
                f"(paired Wilcoxon, alpha {alpha:g})."
            )

    if excluded:
        reasons = "; ".join(f"{m} ({r})" for m, r in excluded.items())
        sentences.append(f"Excluded from this comparison: {reasons}.")
    return sentences


def deployed_model_text(meta: dict, verdict: dict[str, dict]) -> str:
    """
    One line explaining why the saved model was chosen, built from the meta file.

    The meta file gives model_name, selection_rule and wilcoxon_p. The top two models
    and their fit times come from the 7-day verdict. When the meta lacks the rule or
    p-value, or the deployed model is not one of the top two, the line is just the name.
    """
    model = meta["model_name"]
    rule = meta.get("selection_rule")
    p = meta.get("wilcoxon_p")
    v = verdict["7d"]
    top1, top2 = v["best_model"], v["runner_up"]

    if top2 is None or p is None or model not in (top1, top2):
        return f"Deployed model: {model}"

    if rule == "wilcoxon_tie_fit_time":
        fit = v["ranking"].set_index("model")["fit_time_s_mean"]
        other = top2 if model == top1 else top1
        return (
            f"Deployed model: {model}. The top two models by 7-day RMSE ({top1}, {top2}) "
            f"don't differ significantly (p={p:.3f}), so the faster one was chosen "
            f"(fit {fit[model]:.2f} s vs {fit[other]:.1f} s). "
            f"The selection rule uses the 7-day horizon."
        )
    if rule == "wilcoxon_significant" and model == top1:
        return (
            f"Deployed model: {model}. It was chosen because it has the lowest mean 7-day "
            f"RMSE and the difference from {top2} is significant (p={p:.3f})."
        )
    return f"Deployed model: {model}"


def methodology_note(n_origins: int) -> str:
    """One line describing the evaluation design, for the verdict box."""
    return (
        f"Evaluation: {n_origins} monthly origins with an expanding window; 7-day metrics "
        f"use the first {SHORT_HORIZON} steps; the history is under two years; p-values "
        f"are uncorrected for multiple comparisons ({n_origins} origins per test)."
    )


def typical_error_text(metrics: dict, n_origins: int) -> str:
    """
    Mean MAE ± std from one metric block of the meta file, with its origin count.

    `metrics` is a block such as meta["backtest_336_metrics"]. The std is left out
    when it is missing or None, as it is for a single origin.
    """
    mae = metrics["MAE"]
    std = mae.get("std")
    spread = "" if std is None else f" ± {std:.2f}"
    return f"Typical error (MAE): {mae['mean']:.2f}{spread} (backtest, {n_origins} origins)."


def origin_label(origin) -> str:
    """Month label for the origin selector, e.g. 'Mar 2026, model fit up to 2026-03-01'."""
    origin = pd.Timestamp(origin)
    return f"{origin:%b %Y}, model fit up to {origin:%Y-%m-%d}"


# ---------------------------------------------------------------------------
# Table and chart data
# ---------------------------------------------------------------------------


def display_table(horizon_verdict: dict) -> pd.DataFrame:
    """Formatted average table for one horizon, in the order of the verdict ranking."""
    rows = []
    for _, r in horizon_verdict["ranking"].iterrows():
        base = bool(r["is_baseline"])
        rows.append(
            {
                "Model": r["model"],
                "MAE": f"{r['MAE_mean']:.2f}",
                "RMSE (mean ± std)": f"{r['RMSE_mean']:.2f} ± {r['RMSE_std']:.2f}",
                "Wins": "—"
                if base
                else str(int(r["wins"])),  # strings: no mixed-type column
                "Skill vs baseline": "—" if base else f"{r['skill_vs_baseline']:.3f}",
                "Fit time (s)": f"{r['fit_time_s_mean']:.2f}",
                "p-value vs baseline": "—"
                if base
                else _format_p_cell(r["p_vs_baseline"]),
            }
        )
    return pd.DataFrame(rows)


def _format_p_cell(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def origin_table(
    origin_results: pd.DataFrame, origin, baseline: str = BASELINE_NAME
) -> pd.DataFrame:
    """
    Metrics for every model at one origin, for both horizons.

    Failed rows stay in the table with a status message and empty metrics. The
    'Lowest RMSE' column marks the best non-baseline model that succeeded at each horizon.
    """
    rows = origin_results[origin_results["origin"] == pd.Timestamp(origin)].copy()
    if rows.empty:
        raise ValueError(f"No results for origin {pd.Timestamp(origin):%Y-%m-%d}")
    rows["_ok"] = _is_ok(rows)

    out = []
    for horizon in HORIZONS:
        h = rows[rows["horizon"] == horizon]
        base = h[h["model"] == baseline]
        base_rmse = (
            float(base["RMSE"].iloc[0])
            if not base.empty and bool(base["_ok"].iloc[0])
            else np.nan
        )
        cand = h[(h["model"] != baseline) & h["_ok"]]
        best = cand.loc[cand["RMSE"].idxmin(), "model"] if not cand.empty else None
        for _, r in h.sort_values("model").iterrows():
            ok = bool(r["_ok"])
            is_base = r["model"] == baseline
            skill = np.nan
            if ok and not is_base and not np.isnan(base_rmse):
                skill = 1 - r["RMSE"] / base_rmse
            out.append(
                {
                    "Horizon": horizon,
                    "Model": r["model"],
                    "MAE": r["MAE"] if ok else np.nan,
                    "RMSE": r["RMSE"] if ok else np.nan,
                    "MAPE": r["MAPE"] if ok else np.nan,
                    "sMAPE": r["sMAPE"] if ok else np.nan,
                    "Fit time (s)": r["fit_time_s"],
                    "Skill vs baseline": skill,
                    "Lowest RMSE": "Yes" if r["model"] == best else "",
                    "Status": "ok"
                    if ok
                    else (
                        f"failed: {r['error']}"
                        if "error" in r and str(r["error"]).strip()
                        else "failed or missing"
                    ),
                }
            )
    return pd.DataFrame(out).round(
        {
            "MAE": 2,
            "RMSE": 2,
            "MAPE": 3,
            "sMAPE": 3,
            "Fit time (s)": 2,
            "Skill vs baseline": 3,
        }
    )


def rmse_by_origin(origin_results: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """RMSE per origin (rows) and model (columns). Failed origins are NaN, so the line breaks."""
    h = origin_results[origin_results["horizon"] == horizon]
    ok = h[_is_ok(h)]
    return ok.pivot(index="origin", columns="model", values="RMSE").sort_index()


def forecast_slice(
    predictions: pd.DataFrame, origin, models: list[str], horizon: str
) -> pd.DataFrame:
    """
    Actual and forecast values for one origin, sliced to the horizon's steps.

    Returns a frame indexed by timestamp, with an 'actual' column and one column per
    selected model. Empty if no models are selected.
    """
    if not models:
        return pd.DataFrame()
    rows = predictions[
        (predictions["origin"] == pd.Timestamp(origin))
        & predictions["model"].isin(models)
        & (predictions["step"] <= HORIZON_STEPS[horizon])
    ]
    actual = (
        rows.drop_duplicates("timestamp")
        .set_index("timestamp")["y_true"]
        .sort_index()
        .rename("actual")
    )
    forecasts = rows.pivot(
        index="timestamp", columns="model", values="y_pred"
    ).sort_index()
    return pd.concat([actual, forecasts], axis=1)
