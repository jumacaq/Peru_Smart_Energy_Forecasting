import numpy as np
import pandas as pd
import pytest

from src.config import LONG_HORIZON, SHORT_HORIZON
from src.evaluation import (
    make_origins,
    score_forecast,
    select_winner,
    split_at_origin,
    summarize,
)
from src.metrics import evaluate_all

ORIGIN = pd.Timestamp("2025-08-01 00:00")


def _long_frame(rows):
    """Build a long results frame; each row is (origin, model, horizon, rmse, fit_time)."""
    records = []
    for origin, model, horizon, rmse, fit in rows:
        records.append(
            {
                "origin": origin,
                "model": model,
                "horizon": horizon,
                "MAE": rmse * 0.8,
                "RMSE": rmse,
                "MAPE": rmse,
                "sMAPE": rmse,
                "fit_time_s": fit,
            }
        )
    return pd.DataFrame(records)


class TestMakeOrigins:
    def test_twelve_monthly_origins(self):
        origins = make_origins()
        assert len(origins) == 12

    def test_first_and_last_dates(self):
        origins = make_origins()
        assert origins[0] == pd.Timestamp("2025-07-01 00:00")
        assert origins[-1] == pd.Timestamp("2026-06-01 00:00")

    def test_all_at_midnight_on_the_first(self):
        origins = make_origins()
        assert (origins.day == 1).all()
        assert (origins.hour == 0).all()
        assert (origins.minute == 0).all()


class TestSplitAtOrigin:
    @pytest.fixture
    def series(self):
        idx = pd.date_range(ORIGIN - pd.Timedelta(days=10), ORIGIN + pd.Timedelta(days=31), freq="30min")
        return pd.Series(np.arange(len(idx), dtype=float), index=idx, name="Demand")

    def test_train_ends_at_origin_inclusive(self, series):
        train, _ = split_at_origin(series, ORIGIN)
        assert train.index[-1] == ORIGIN
        assert train.index[0] == series.index[0]

    def test_test_starts_30_min_later_and_has_1440_steps(self, series):
        _, test = split_at_origin(series, ORIGIN)
        assert test.index[0] == ORIGIN + pd.Timedelta(minutes=30)
        assert len(test) == LONG_HORIZON == 1440
        assert test.index[-1] == ORIGIN + pd.Timedelta(minutes=30 * LONG_HORIZON)

    def test_train_and_test_do_not_overlap(self, series):
        train, test = split_at_origin(series, ORIGIN)
        assert train.index.max() < test.index.min()

    def test_incomplete_test_window_raises(self, series):
        short = series[series.index <= ORIGIN + pd.Timedelta(days=20)]
        with pytest.raises(ValueError, match="incomplete"):
            split_at_origin(short, ORIGIN)

    def test_requires_datetime_index(self):
        with pytest.raises(TypeError, match="DatetimeIndex"):
            split_at_origin(pd.Series([1.0, 2.0, 3.0]), ORIGIN)


class TestScoreForecast:
    def test_returns_7d_and_30d_metrics(self):
        y_true = np.zeros(LONG_HORIZON)
        y_pred = np.zeros(LONG_HORIZON)
        scores = score_forecast(y_true, y_pred)

        assert set(scores) == {"7d", "30d"}
        assert set(scores["7d"]) == {"MAE", "RMSE", "MAPE", "sMAPE"}

    def test_7d_uses_first_336_steps_only(self):
        y_true = np.full(LONG_HORIZON, 10.0)
        y_pred = y_true.copy()
        y_pred[SHORT_HORIZON:] += 1.0  # errors only after day 7

        scores = score_forecast(y_true, y_pred)

        assert scores["7d"]["RMSE"] == pytest.approx(0.0)
        assert scores["30d"]["RMSE"] == pytest.approx(
            np.sqrt((LONG_HORIZON - SHORT_HORIZON) / LONG_HORIZON)
        )
        assert scores["30d"] == evaluate_all(y_true, y_pred)

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="Expected"):
            score_forecast(np.zeros(100), np.zeros(100))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            score_forecast(np.zeros(LONG_HORIZON), np.zeros(LONG_HORIZON - 1))


ORIGINS = list(pd.date_range("2025-07-01", periods=3, freq="MS"))


class TestSummarize:
    @pytest.fixture
    def results(self):
        rows = []
        for i, origin in enumerate(ORIGINS):
            for horizon in ["7d", "30d"]:
                rows.append((origin, "SeasonalNaive", horizon, 10.0, 0.1))
                rows.append((origin, "A", horizon, 5.0, 1.0))
                b_rmse = 4.0 if (horizon == "7d" and i == 2) else 8.0
                rows.append((origin, "B", horizon, b_rmse, 2.0))
        return _long_frame(rows)

    def _row(self, summary, model, horizon):
        return summary[(summary["model"] == model) & (summary["horizon"] == horizon)].iloc[0]

    def test_one_row_per_model_and_horizon(self, results):
        summary = summarize(results)
        assert len(summary) == 6

    def test_rmse_mean_std_median(self, results):
        summary = summarize(results)
        a_7d = self._row(summary, "A", "7d")
        assert a_7d["RMSE_mean"] == pytest.approx(5.0)
        assert a_7d["RMSE_std"] == pytest.approx(0.0)
        assert a_7d["RMSE_median"] == pytest.approx(5.0)

        b_7d = self._row(summary, "B", "7d")
        assert b_7d["RMSE_mean"] == pytest.approx((8 + 8 + 4) / 3)
        assert b_7d["RMSE_median"] == pytest.approx(8.0)

    def test_mean_fit_time(self, results):
        summary = summarize(results)
        assert self._row(summary, "A", "7d")["fit_time_s_mean"] == pytest.approx(1.0)
        assert self._row(summary, "B", "7d")["fit_time_s_mean"] == pytest.approx(2.0)

    def test_wins_count_lowest_rmse_excluding_baseline(self, results):
        summary = summarize(results)
        # 7d: A wins origins 1 and 2; B wins origin 3 (RMSE 4 < 5).
        assert self._row(summary, "A", "7d")["wins"] == 2
        assert self._row(summary, "B", "7d")["wins"] == 1
        # 30d: A wins every origin.
        assert self._row(summary, "A", "30d")["wins"] == 3
        assert self._row(summary, "B", "30d")["wins"] == 0
        assert self._row(summary, "SeasonalNaive", "7d")["wins"] == 0

    def test_skill_vs_baseline_is_mean_of_per_origin_skill(self, results):
        summary = summarize(results)
        # A: 1 - 5/10 = 0.5 on every origin.
        assert self._row(summary, "A", "7d")["skill_vs_baseline"] == pytest.approx(0.5)
        # B 7d: per origin 0.2, 0.2, 0.6 -> mean 1/3.
        assert self._row(summary, "B", "7d")["skill_vs_baseline"] == pytest.approx(1 / 3)
        assert self._row(summary, "SeasonalNaive", "7d")["skill_vs_baseline"] == pytest.approx(0.0)

    def test_missing_baseline_raises(self, results):
        no_baseline = results[results["model"] != "SeasonalNaive"]
        with pytest.raises(ValueError, match="Baseline"):
            summarize(no_baseline)


class TestSelectWinner:
    def test_clear_winner_by_significant_wilcoxon(self):
        origins = make_origins()
        rows = []
        for i, origin in enumerate(origins):
            rows.append((origin, "SeasonalNaive", "7d", 10.0, 0.1))
            rows.append((origin, "A", "7d", 5.0 + 0.1 * i, 2.0))  # slower
            rows.append((origin, "B", "7d", 8.0 + 0.1 * i, 0.5))  # faster, but worse
        result = select_winner(_long_frame(rows))

        assert result["winner"] == "A"
        assert result["runner_up"] == "B"
        assert result["rule"] == "wilcoxon_significant"
        assert result["p_value"] < 0.05
        assert list(result["ranking"]["model"]) == ["A", "B"]
        assert list(result["ranking"]["rank"]) == [1, 2]

    def test_tie_rule_picks_faster_model(self):
        origins = make_origins()
        # Paired differences (B - A) have mixed signs, so the Wilcoxon test
        # finds no significant difference. A has the lower mean RMSE by a small margin.
        diffs = [0.3, -0.3, 0.2, -0.2, 0.1, -0.1, 0.3, -0.3, 0.2, -0.2, 0.1, -0.05]
        rows = []
        for i, origin in enumerate(origins):
            a_rmse = 5.0 + 0.1 * i
            rows.append((origin, "SeasonalNaive", "7d", 10.0, 0.1))
            rows.append((origin, "A", "7d", a_rmse, 2.0))  # slower
            rows.append((origin, "B", "7d", a_rmse + diffs[i], 1.0))  # faster
        result = select_winner(_long_frame(rows))

        assert result["ranking"].iloc[0]["model"] == "A"
        assert result["p_value"] >= 0.05
        assert result["rule"] == "wilcoxon_tie_fit_time"
        assert result["winner"] == "B"
        assert result["runner_up"] == "A"

    def test_single_model(self):
        origins = make_origins()
        rows = []
        for i, origin in enumerate(origins):
            rows.append((origin, "SeasonalNaive", "7d", 10.0, 0.1))
            rows.append((origin, "A", "7d", 5.0 + 0.1 * i, 2.0))
        result = select_winner(_long_frame(rows))

        assert result["winner"] == "A"
        assert result["runner_up"] is None
        assert result["p_value"] is None
        assert result["rule"] == "single_model"
        assert len(result["ranking"]) == 1

    def test_no_models_raises(self):
        origins = make_origins()
        rows = [(origin, "SeasonalNaive", "7d", 10.0, 0.1) for origin in origins]
        with pytest.raises(ValueError, match="No non-baseline"):
            select_winner(_long_frame(rows))
