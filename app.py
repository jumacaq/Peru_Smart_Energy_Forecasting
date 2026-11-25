"""
Streamlit app: forecast electricity demand with the saved best model, and backtest it.

Run with:  uv run streamlit run app.py
"""

import importlib
import json
import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import BEST_MODEL_META_PATH, BEST_MODEL_PATH, DEMAND_PARQUET_PATH
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
        "Forecast horizon (applies to both tabs)",
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
tab_forecast, tab_backtest = st.tabs(["Forecast", "Backtest"])

with tab_forecast:
    last_observed = demand.index[-1]
    forecast = load_best_model().predict(horizon=steps)

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
        title=f"Last 7 days of actual demand and {horizon_hours}-hour forecast",
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
        file_name=f"forecast_{horizon_hours}h.csv",
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
