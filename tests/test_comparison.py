import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.comparison import (
    build_verdict,
    deployed_model_text,
    display_table,
    forecast_slice,
    methodology_note,
    origin_table,
    rmse_by_origin,
    typical_error_text,
    verdict_text,
)
from src.config import LONG_HORIZON, SHORT_HORIZON
from src.evaluation import select_winner

ROOT = Path(__file__).resolve().parents[1]
BASE = "SeasonalNaive"
ORIGINS = pd.date_range("2025-07-01", periods=12, freq="MS")
IDX = np.arange(12, dtype=float)


def make_results(curves, failed=None, dropped=None, fit_times=None):
    """
    Long results frame from per-origin 7-day RMSE curves, one array of 12 values per model.

    The 30-day RMSE is the 7-day value times 1.2, so both horizons order models the same way.
    `failed` maps model -> origin indexes where the run failed (NaN metrics, error text).
    `dropped` maps model -> origin indexes with no row at all.
    `fit_times` maps model -> fit time in seconds (default 1.0).
    """
    failed = failed or {}
    dropped = dropped or {}
    fit_times = fit_times or {}
    rows = []
    for model, curve in curves.items():
        for i, origin in enumerate(ORIGINS):
            if i in dropped.get(model, []):
                continue
            bad = i in failed.get(model, [])
            for horizon, scale in (("7d", 1.0), ("30d", 1.2)):
                rmse = float(curve[i]) * scale
                rows.append(
                    {
                        "origin": origin,
                        "model": model,
                        "horizon": horizon,
                        "MAE": np.nan if bad else rmse * 0.8,
                        "RMSE": np.nan if bad else rmse,
                        "MAPE": np.nan if bad else rmse / 10,
                        "sMAPE": np.nan if bad else rmse / 10,
                        "fit_time_s": np.nan if bad else fit_times.get(model, 1.0),
                        "error": "RuntimeError: boom" if bad else "",
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def clear_winner_curves():
    a = 150 + IDX  # best model
    b = a + 10 + 0.5 * IDX  # consistently worse, distinct differences
    base = a + 60  # baseline is clearly worse than both
    return {"A": a, "B": b, BASE: base}


def test_clear_winner_is_significant(clear_winner_curves):
    verdict = build_verdict(make_results(clear_winner_curves))
    v = verdict["7d"]

    assert v["best_model"] == "A"
    assert v["runner_up"] == "B"
    assert v["top_two_p_value"] < 0.05
    assert v["clear_winner"] is True
    assert set(v["significant_vs_baseline"]) == {"A", "B"}
    assert v["n_origins"] == 12

    text = " ".join(verdict_text(verdict))
    assert "A has the lowest mean RMSE at 7 days" in text
    assert "statistically significant" in text
    assert "no clear winner" not in text


def test_tie_gives_no_clear_winner():
    same = 150 + IDX
    verdict = build_verdict(
        make_results({"A": same, "B": same.copy(), BASE: same + 50})
    )
    v = verdict["7d"]

    assert v["top_two_p_value"] == 1.0
    assert v["clear_winner"] is False
    text = " ".join(verdict_text(verdict))
    assert "no clear winner" in text
    assert "p=1.000" in text


def test_baseline_not_beaten_is_reported():
    base = 100 + 2 * IDX
    # Both models are consistently worse than the baseline. The test is significant
    # (p < 0.05) but in the wrong direction, so neither model counts as beating it.
    worse_a = base + 5 + 0.1 * IDX
    worse_b = base + 3 + 0.3 * IDX
    verdict = build_verdict(make_results({"A": worse_a, "B": worse_b, BASE: base}))
    v = verdict["7d"]

    assert v["significant_vs_baseline"] == []
    ranking = v["ranking"].set_index("model")
    assert ranking.loc["A", "p_vs_baseline"] < 0.05
    assert bool(ranking.loc["A", "significant_vs_baseline"]) is False

    text = " ".join(verdict_text(verdict))
    assert f"No model significantly beats the {BASE} baseline at 7 days." in text


def test_failed_model_is_excluded_and_reported(clear_winner_curves):
    curves = {**clear_winner_curves, "C": 1 + IDX}  # C would otherwise rank first
    results = make_results(curves, failed={"C": [3]})
    verdict = build_verdict(results)

    for horizon in ("7d", "30d"):
        v = verdict[horizon]
        assert set(v["excluded"]) == {"C"}
        assert "C" not in set(v["ranking"]["model"])
        assert v["best_model"] == "A"

    text = " ".join(verdict_text(verdict))
    assert (
        "Excluded from this comparison: C (failed or missing at 1 of 12 origins)."
        in text
    )


def test_missing_rows_count_as_missing():
    curves = {"A": 150 + IDX, "B": 160 + IDX, BASE: 200 + IDX}
    verdict = build_verdict(make_results(curves, dropped={"B": [0, 5]}))
    assert verdict["7d"]["excluded"] == {"B": "failed or missing at 2 of 12 origins"}


def test_single_model_has_no_pairwise_test():
    verdict = build_verdict(make_results({"A": 150 + IDX, BASE: 200 + IDX}))
    v = verdict["7d"]
    assert v["runner_up"] is None
    assert v["top_two_p_value"] is None
    assert v["clear_winner"] is False
    assert "only model" in " ".join(verdict_text(verdict))


def test_baseline_failure_raises(clear_winner_curves):
    results = make_results(clear_winner_curves, failed={BASE: [2]})
    with pytest.raises(ValueError, match="Baseline 'SeasonalNaive'"):
        build_verdict(results)


def test_no_candidate_models_raises():
    results = make_results({BASE: 200 + IDX, "A": 150 + IDX}, failed={"A": [0]})
    with pytest.raises(ValueError, match="No non-baseline model"):
        build_verdict(results)


def test_top_two_p_matches_select_winner(clear_winner_curves):
    results = make_results(clear_winner_curves)
    verdict = build_verdict(results)
    expected = select_winner(results[results["horizon"] == "7d"])
    assert verdict["7d"]["top_two_p_value"] == pytest.approx(expected["p_value"])


def test_30d_uses_30d_rows(clear_winner_curves):
    verdict = build_verdict(make_results(clear_winner_curves))
    ranking_7 = verdict["7d"]["ranking"].set_index("model")
    ranking_30 = verdict["30d"]["ranking"].set_index("model")
    assert ranking_30.loc["A", "RMSE_mean"] == pytest.approx(
        1.2 * ranking_7.loc["A", "RMSE_mean"]
    )


def test_ranking_has_worst_origin_and_baseline_row(clear_winner_curves):
    ranking = build_verdict(make_results(clear_winner_curves))["7d"]["ranking"]
    assert list(ranking["model"])[-1] == BASE
    assert bool(ranking.iloc[-1]["is_baseline"]) is True
    a = ranking.set_index("model").loc["A"]
    assert a["RMSE_worst"] == pytest.approx(150 + 11)
    assert a["rank"] == 1


def test_methodology_note_mentions_design():
    note = methodology_note(12)
    assert "12 monthly origins" in note
    assert f"first {SHORT_HORIZON} steps" in note
    assert "uncorrected" in note


def test_display_table_columns_and_baseline_row(clear_winner_curves):
    table = display_table(build_verdict(make_results(clear_winner_curves))["7d"])
    assert list(table.columns) == [
        "Model",
        "MAE",
        "RMSE (mean ± std)",
        "Wins",
        "Skill vs baseline",
        "Fit time (s)",
        "p-value vs baseline",
    ]
    base_row = table[table["Model"] == BASE].iloc[0]
    assert base_row["p-value vs baseline"] == "—"


def test_display_tables_are_arrow_serialisable(clear_winner_curves):
    # st.dataframe fails on mixed int/str columns, so every display cell must be one type.
    pa = pytest.importorskip("pyarrow")
    results = make_results(clear_winner_curves, failed={"A": [2]})
    verdict = build_verdict(make_results(clear_winner_curves))
    pa.Table.from_pandas(display_table(verdict["7d"]), preserve_index=False)
    pa.Table.from_pandas(origin_table(results, ORIGINS[2]), preserve_index=False)


def test_origin_table_marks_failures_and_best(clear_winner_curves):
    curves = {**clear_winner_curves, "C": 1 + IDX}
    results = make_results(curves, failed={"C": [3]})
    table = origin_table(results, ORIGINS[3])
    c_rows = table[table["Model"] == "C"]
    assert set(c_rows["Status"].str.startswith("failed")) == {True}
    best_rows = table[table["Lowest RMSE"] == "Yes"]
    assert set(best_rows["Model"]) == {"A"}
    assert len(table) == 2 * 4  # 2 horizons x 4 models


def test_rmse_by_origin_leaves_failures_empty(clear_winner_curves):
    results = make_results(clear_winner_curves, failed={"B": [4]})
    wide = rmse_by_origin(results, "7d")
    assert wide.shape == (12, 3)
    assert np.isnan(wide.loc[ORIGINS[4], "B"])


def test_forecast_slice_limits_steps_to_horizon():
    rows = []
    stamps = pd.date_range("2025-07-01 00:30", periods=LONG_HORIZON, freq="30min")
    for model in ("A", "B"):
        for step, ts in enumerate(stamps, start=1):
            rows.append(
                {
                    "origin": ORIGINS[0],
                    "model": model,
                    "timestamp": ts,
                    "step": step,
                    "y_true": float(step),
                    "y_pred": float(step) + (1 if model == "A" else 2),
                }
            )
    preds = pd.DataFrame(rows)

    short = forecast_slice(preds, ORIGINS[0], ["A", "B"], "7d")
    assert len(short) == SHORT_HORIZON
    assert list(short.columns) == ["actual", "A", "B"]

    long = forecast_slice(preds, ORIGINS[0], ["A"], "30d")
    assert len(long) == LONG_HORIZON
    assert (long["A"] - long["actual"] == 1).all()

    assert forecast_slice(preds, ORIGINS[0], [], "7d").empty


def test_comparison_module_does_not_import_torch_or_prophet():
    code = "import sys, src.comparison; assert 'torch' not in sys.modules and 'prophet' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_deployed_model_text_tie_uses_faster_model():
    same = 150 + IDX
    results = make_results(
        {"A": same, "B": same.copy(), BASE: same + 50},
        fit_times={"A": 5.0, "B": 0.5},
    )
    verdict = build_verdict(results)
    meta = {
        "model_name": "B",
        "selection_rule": "wilcoxon_tie_fit_time",
        "wilcoxon_p": 1.0,
    }
    assert deployed_model_text(meta, verdict) == (
        "Deployed model: B. The top two models by 7-day RMSE (A, B) don't differ "
        "significantly (p=1.000), so the faster one was chosen (fit 0.50 s vs 5.0 s). "
        "The selection rule uses the 7-day horizon."
    )


def test_deployed_model_text_significant_case(clear_winner_curves):
    verdict = build_verdict(make_results(clear_winner_curves))
    meta = {
        "model_name": "A",
        "selection_rule": "wilcoxon_significant",
        "wilcoxon_p": 0.001,
    }
    assert deployed_model_text(meta, verdict) == (
        "Deployed model: A. It was chosen because it has the lowest mean 7-day RMSE and "
        "the difference from B is significant (p=0.001)."
    )


def test_deployed_model_text_missing_keys_falls_back(clear_winner_curves):
    verdict = build_verdict(make_results(clear_winner_curves))
    assert deployed_model_text({"model_name": "A"}, verdict) == "Deployed model: A"
    partial = {"model_name": "A", "selection_rule": "wilcoxon_significant"}
    assert deployed_model_text(partial, verdict) == "Deployed model: A"


def test_typical_error_text_with_and_without_std():
    block = {"MAE": {"mean": 196.7301, "std": 73.7468}}
    assert typical_error_text(block, 12) == (
        "Typical error (MAE): 196.73 ± 73.75 (backtest, 12 origins)."
    )
    assert typical_error_text({"MAE": {"mean": 196.73, "std": None}}, 1) == (
        "Typical error (MAE): 196.73 (backtest, 1 origins)."
    )
    assert typical_error_text({"MAE": {"mean": 196.73}}, 12) == (
        "Typical error (MAE): 196.73 (backtest, 12 origins)."
    )
