"""
Streamlit app: forecast electricity demand with the saved best model, backtest it,
and compare the benchmark models.

The comparison tab reads the benchmark files in reports/ and never imports the model
classes, so torch and prophet are not loaded by it.

Run with:  uv run streamlit run app.py
"""

import importlib
import json
import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import qualitative

from src.comparison import (
    build_verdict,
    deployed_model_text,
    display_table,
    forecast_slice,
    methodology_note,
    origin_label,
    origin_table,
    rmse_by_origin,
    typical_error_text,
    verdict_text,
)
from src.config import (
    BASELINE_NAME,
    BENCHMARK_ORIGINS_CSV,
    BEST_MODEL_META_PATH,
    BEST_MODEL_PATH,
    DEMAND_PARQUET_PATH,
    LONG_HORIZON,
    PREDICTIONS_PARQUET,
    ROOT,
    SHORT_HORIZON,
)
from src.data_loader import load_processed_dataset
from src.metrics import evaluate_all

st.set_page_config(page_title="Electricity demand forecast (Peru)", layout="wide")

# Data is 30-minute resolution, so 1 hour = 2 steps.
STEPS_PER_HOUR = 2
HORIZON_HOURS = list(range(24, 169, 24))  # 24, 48, ..., 168
TIME_SLOTS = [t.strftime("%H:%M") for t in pd.date_range("2000-01-01", periods=48, freq="30min")]

# One week of 30-minute steps. Used as the minimum history (the model has a lag-336
# feature) and as the minimum amount of data after a backtest origin.
ONE_WEEK_STEPS = 336


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_meta() -> dict:
    return json.loads(BEST_MODEL_META_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_demand() -> pd.Series:
    return load_processed_dataset(DEMAND_PARQUET_PATH)["Demand"]


def model_class(meta: dict):
    """Import the forecaster class named in the meta file (imported only when needed)."""
    module_name, class_name = meta["class_path"].rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


@st.cache_resource
def load_best_model():
    meta = load_meta()
    return model_class(meta).load(str(BEST_MODEL_PATH))


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Refitting the model on data up to the origin...")
def run_backtest(origin: str, steps: int) -> tuple[pd.DataFrame, dict]:
    """
    Refit a fresh model on data up to and including `origin`, then forecast the next
    `steps` steps. Mirrors run_one() in scripts/run_benchmark.py: same class, default
    constructor arguments, same seed, same train/test split.
    """
    meta = load_meta()
    demand = load_demand()

    pos = demand.index.get_loc(pd.Timestamp(origin))
    train = demand.iloc[: pos + 1]
    actual = demand.iloc[pos + 1 : pos + 1 + steps]

    random.seed(meta["seed"])
    np.random.seed(meta["seed"])
    model = model_class(meta)()
    model.fit(train)
    pred = model.predict(horizon=steps)
    if not pred.index.equals(actual.index):
        raise ValueError("Forecast index does not match the actual index.")

    metrics = evaluate_all(actual, pred)
    frame = pd.DataFrame({"actual": actual, "forecast": pred.to_numpy()}, index=actual.index)
    return frame, metrics


# ---------------------------------------------------------------------------
# Model comparison (reads reports/ only; no model classes are imported here)
# ---------------------------------------------------------------------------

BENCHMARK_COMMAND = "uv run python scripts/run_benchmark.py"
ALL_ORIGINS = "All origins (average)"
HORIZON_OPTIONS = ["7d", "30d"]  # labels used in reports/
MODEL_PALETTE = qualitative.Plotly


def _file_stamp(path) -> float:
    """File modification time. Passed to the cached loaders so a regenerated file reloads."""
    return path.stat().st_mtime if path.exists() else 0.0


FORECAST_HORIZONS = {"7 days": SHORT_HORIZON, "30 days": LONG_HORIZON}  # 336 and 1440 steps
# Metric block in the meta file that holds the typical error for each forecast horizon.
ERROR_BLOCKS = {"7 days": "backtest_336_metrics", "30 days": "holdout_1440_metrics"}


@st.cache_data
def load_benchmark_verdict(path: str, stamp: float) -> tuple[pd.DataFrame, dict]:
    """Benchmark rows and the verdict built from them. Shared by the Forecast and comparison tabs."""
    df = pd.read_csv(path, parse_dates=["origin"])
    df["error"] = df["error"].fillna("")
    return df, build_verdict(df)


def benchmark_verdict() -> tuple[pd.DataFrame, dict] | None:
    """(results, verdict), or None when the benchmark file is missing. Raises ValueError if unscorable."""
    if not BENCHMARK_ORIGINS_CSV.exists():
        return None
    return load_benchmark_verdict(str(BENCHMARK_ORIGINS_CSV), _file_stamp(BENCHMARK_ORIGINS_CSV))


@st.cache_data
def load_predictions(path: str, stamp: float) -> pd.DataFrame:
    return pd.read_parquet(path)


def model_colors(models) -> dict[str, str]:
    """One colour per model, assigned by sorted name so each model keeps its colour."""
    return {m: MODEL_PALETTE[i % len(MODEL_PALETTE)] for i, m in enumerate(sorted(models))}


def _line_style(model: str, color: str) -> dict:
    return {"color": color, "width": 2, "dash": "dash" if model == BASELINE_NAME else "solid"}


def month_chart(rmse: pd.DataFrame, colors: dict[str, str], horizon: str) -> go.Figure:
    fig = go.Figure()
    for model in rmse.columns:
        fig.add_trace(go.Scatter(
            x=rmse.index, y=rmse[model], mode="lines+markers", name=model,
            line=_line_style(model, colors[model]),
        ))
    fig.update_layout(
        title=f"RMSE by forecast origin ({horizon})",
        xaxis_title="Forecast origin",
        yaxis_title="RMSE",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        height=450,
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
    )
    return fig


def forecast_chart(wide: pd.DataFrame, colors: dict[str, str], origin, horizon: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wide.index, y=wide["actual"], mode="lines", name="Actual",
        line={"color": "#333333", "width": 2},
    ))
    for model in wide.columns:
        if model == "actual":
            continue
        fig.add_trace(go.Scatter(
            x=wide.index, y=wide[model], mode="lines", name=model,
            line=_line_style(model, colors[model]),
        ))
    fig.update_layout(
        title=f"Actual vs forecast from origin {pd.Timestamp(origin):%Y-%m-%d} ({horizon})",
        xaxis_title="Timestamp",
        yaxis_title="Demand",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        height=450,
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
    )
    return fig


def render_average(results: pd.DataFrame, verdict: dict, colors: dict[str, str]) -> None:
    st.subheader("Average across origins")
    horizon = st.radio("Horizon", HORIZON_OPTIONS, horizontal=True, key="comparison_avg_horizon")
    st.dataframe(display_table(verdict[horizon]), hide_index=True)

    st.subheader("RMSE by forecast origin")
    chart_horizon = st.radio(
        "Horizon", HORIZON_OPTIONS, horizontal=True, key="comparison_chart_horizon"
    )
    st.plotly_chart(month_chart(rmse_by_origin(results, chart_horizon), colors, chart_horizon))


def render_single_origin(
    results: pd.DataFrame, preds: pd.DataFrame, origin: pd.Timestamp, meta: dict, colors: dict[str, str]
) -> None:
    st.subheader(f"Metrics for {origin_label(origin)}")
    st.dataframe(origin_table(results, origin), hide_index=True)

    st.subheader("Forecast vs actual")
    available = sorted(preds["model"].unique())
    default = [m for m in dict.fromkeys([meta["model_name"], BASELINE_NAME]) if m in available]
    selected = st.multiselect("Models", available, default=default, key="comparison_models")
    horizon = st.radio("Horizon", HORIZON_OPTIONS, horizontal=True, key="comparison_fc_horizon")

    wide = forecast_slice(preds, origin, selected, horizon)
    if wide.empty:
        st.info("Select at least one model with a forecast at this origin.")
        return
    st.plotly_chart(forecast_chart(wide, colors, origin, horizon))


def render_comparison_tab(meta: dict) -> None:
    missing = [p for p in (BENCHMARK_ORIGINS_CSV, PREDICTIONS_PARQUET) if not p.exists()]
    if missing:
        names = ", ".join(f"`{p.relative_to(ROOT).as_posix()}`" for p in missing)
        st.info(f"Benchmark files not found: {names}. Generate them with `{BENCHMARK_COMMAND}`.")
        return

    preds = load_predictions(str(PREDICTIONS_PARQUET), _file_stamp(PREDICTIONS_PARQUET))
    try:
        results, verdict = benchmark_verdict()
    except ValueError as exc:
        st.error(f"Model comparison unavailable: {exc}")
        return
    colors = model_colors(results["model"].unique())
    origins = [pd.Timestamp(o) for o in sorted(results["origin"].unique())]

    choice = st.selectbox(
        "Origin",
        [ALL_ORIGINS, *origins],
        format_func=lambda o: o if isinstance(o, str) else origin_label(o),
        key="comparison_origin",
    )
    st.caption(
        "Each month uses a model refit on data up to that date; "
        "the Forecast tab uses the final model."
    )

    # The verdict always uses the averages across origins, whatever the selector shows.
    with st.container(border=True):
        st.markdown("**Verdict** (averages across origins)")
        st.markdown("\n".join(f"- {s}" for s in verdict_text(verdict)))
        st.markdown(deployed_model_text(meta, verdict))
        st.caption(methodology_note(verdict["7d"]["n_origins"]))

    if choice == ALL_ORIGINS:
        render_average(results, verdict, colors)
        st.subheader("Forecast vs actual")
        st.info("Select a month above to see forecast vs actual.")
    else:
        render_single_origin(results, preds, choice, meta, colors)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

meta = load_meta()
demand = load_demand()

with st.sidebar:
    st.header("Model")
    st.markdown(f"**{meta['model_name']}**")
    st.caption(meta["class_path"])

    st.subheader(f"Backtest ({meta['n_origins']} origins × 7 days)")
    bt = meta["backtest_336_metrics"]
    for key, label, unit in [("MAE", "MAE", ""), ("RMSE", "RMSE", ""),
                             ("MAPE", "MAPE", " %"), ("sMAPE", "sMAPE", " %")]:
        st.metric(
            label,
            f"{bt[key]['mean']:.2f}{unit}",
            delta=f"± {bt[key]['std']:.2f}",
            delta_color="off",
        )

    st.subheader("Training data")
    tr = meta["training_date_range"]
    st.write(f"From **{tr['start']}**")
    st.write(f"To **{tr['end']}**")
    st.write(f"{tr['n_rows']:,} rows (30-minute steps)")

    st.subheader("Horizon")
    horizon_hours = st.selectbox(
        "Backtest horizon",
        HORIZON_HOURS,
        index=0,
        format_func=lambda h: f"{h} h ({h * STEPS_PER_HOUR} steps)",
        key="horizon_hours",
    )
    steps = horizon_hours * STEPS_PER_HOUR


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

st.title("Electricity demand forecast (Peru)")
tab_forecast, tab_backtest, tab_comparison = st.tabs(["Forecast", "Backtest", "Model comparison"])

with tab_forecast:
    forecast_label = st.radio(
        "Forecast horizon", list(FORECAST_HORIZONS), horizontal=True, key="forecast_horizon",
    )
    forecast_steps = FORECAST_HORIZONS[forecast_label]
    st.caption(typical_error_text(meta[ERROR_BLOCKS[forecast_label]], meta["n_origins"]))

    try:
        bench = benchmark_verdict()
    except ValueError:
        bench = None  # the comparison tab reports the error; here the caption is skipped
    if (
        forecast_label == "30 days"
        and bench is not None
        and not bench[1]["30d"]["significant_vs_baseline"]
    ):
        st.caption(
            "At 30 days, accuracy is not significantly better than the seasonal-naive "
            "baseline; see the Model comparison tab."
        )

    last_observed = demand.index[-1]
    forecast = load_best_model().predict(horizon=forecast_steps)

    st.info(
        f"The forecast starts after the last observed timestamp in the data "
        f"({last_observed:%Y-%m-%d %H:%M}), not from today's date. "
        f"The first forecast timestamp is {forecast.index[0]:%Y-%m-%d %H:%M}."
    )

    recent = demand[demand.index > last_observed - pd.Timedelta(days=7)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent.to_numpy(), mode="lines", name="Actual",
        line={"color": "#1f77b4", "width": 2},
    ))
    fig.add_trace(go.Scatter(
        x=forecast.index, y=forecast.to_numpy(), mode="lines", name="Forecast",
        line={"color": "#ff7f0e", "width": 2, "dash": "dash"},
    ))
    fig.update_layout(
        title=f"Last 7 days of actual demand and {forecast_label} forecast",
        xaxis_title="Timestamp",
        yaxis_title="Demand",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        height=450,
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
    )
    st.plotly_chart(fig)

    table = pd.DataFrame({
        "timestamp": forecast.index,
        "forecast_demand": forecast.to_numpy().round(2),
    })
    st.dataframe(table, hide_index=True)
    st.download_button(
        "Download forecast as CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=f"forecast_{forecast_steps}steps.csv",
        mime="text/csv",
    )

with tab_backtest:
    first_origin = demand.index[ONE_WEEK_STEPS - 1]        # at least one week of history
    last_origin = demand.index[-(ONE_WEEK_STEPS + 1)]      # at least one week after it

    st.write(
        f"Choose an origin between **{first_origin:%Y-%m-%d %H:%M}** and "
        f"**{last_origin:%Y-%m-%d %H:%M}**. The model is refit on all data up to and "
        f"including the origin, then forecasts the next {steps} steps "
        f"({horizon_hours} h) and is compared with the actual demand."
    )

    col_date, col_time = st.columns(2)
    with col_date:
        origin_date = st.date_input(
            "Origin date",
            value=last_origin.date(),
            min_value=first_origin.date(),
            max_value=last_origin.date(),
            key="origin_date",
        )
    with col_time:
        origin_time = st.selectbox("Origin time", TIME_SLOTS, index=0, key="origin_time")

    origin = pd.Timestamp(f"{origin_date} {origin_time}")
    if not first_origin <= origin <= last_origin:
        st.error(
            f"Origin must be between {first_origin:%Y-%m-%d %H:%M} and "
            f"{last_origin:%Y-%m-%d %H:%M}."
        )
    else:
        frame, metrics = run_backtest(origin.isoformat(), steps)

        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{metrics['MAE']:.2f}")
        c2.metric("RMSE", f"{metrics['RMSE']:.2f}")
        c3.metric("MAPE", f"{metrics['MAPE']:.2f} %")

        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(
            x=frame.index, y=frame["actual"], mode="lines", name="Actual",
            line={"color": "#1f77b4", "width": 2},
        ))
        fig_bt.add_trace(go.Scatter(
            x=frame.index, y=frame["forecast"], mode="lines", name="Backtest forecast",
            line={"color": "#ff7f0e", "width": 2, "dash": "dash"},
        ))
        fig_bt.update_layout(
            title=f"Backtest from origin {origin:%Y-%m-%d %H:%M} ({horizon_hours} h)",
            xaxis_title="Timestamp",
            yaxis_title="Demand",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
            height=450,
            margin={"l": 10, "r": 10, "t": 60, "b": 10},
        )
        st.plotly_chart(fig_bt)
        st.caption(
            f"Training data ends at {origin:%Y-%m-%d %H:%M}. Test window: "
            f"{frame.index[0]:%Y-%m-%d %H:%M} to {frame.index[-1]:%Y-%m-%d %H:%M}."
        )

with tab_comparison:
    render_comparison_tab(meta)
