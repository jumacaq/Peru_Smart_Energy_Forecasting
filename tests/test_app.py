import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
ALL_ORIGINS = "All origins (average)"


def _run_app() -> AppTest:
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=300).run()


def test_app_runs_and_comparison_tab_renders():
    at = _run_app()

    assert not at.exception
    assert [t.label for t in at.tabs] == ["Forecast", "Backtest", "Model comparison"]
    assert any("Verdict" in m.value for m in at.markdown)
    origin_select = at.selectbox(key="comparison_origin")
    assert origin_select.value == ALL_ORIGINS


def test_comparison_tab_single_origin_renders():
    at = _run_app()
    origin_select = at.selectbox(key="comparison_origin")
    origin_select.select_index(1).run()

    assert not at.exception
    assert at.multiselect(key="comparison_models").value
    assert any("Metrics for" in h.value for h in at.subheader)


def test_verdict_box_shows_deployed_model_line():
    at = _run_app()
    assert not at.exception
    assert any(m.value.startswith("Deployed model: ") for m in at.markdown)


def _forecast_chart_x_lengths(at: AppTest) -> list[int]:
    """Number of points in each trace of the first chart on the page (the Forecast tab's)."""
    spec = json.loads(at.get("plotly_chart")[0].proto.spec)
    return [len(trace["x"]) for trace in spec["data"]]


def test_forecast_horizon_toggle():
    at = _run_app()
    assert not at.exception
    assert at.radio(key="forecast_horizon").value == "7 days"
    assert len(at.dataframe[0].value) == 336
    assert _forecast_chart_x_lengths(at)[1] == 336
    assert any(c.value.startswith("Typical error (MAE): ") and "backtest, 12 origins" in c.value
               for c in at.caption)
    assert not any("At 30 days" in c.value for c in at.caption)

    at.radio(key="forecast_horizon").set_value("30 days").run()
    assert not at.exception
    assert len(at.dataframe[0].value) == 1440
    assert _forecast_chart_x_lengths(at)[1] == 1440
    # Data-dependent: holds while no model significantly beats the baseline at 30 days.
    assert any("At 30 days, accuracy is not significantly better" in c.value for c in at.caption)
