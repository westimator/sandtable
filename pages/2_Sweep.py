"""
Page 2: Parameter Sweep

Run a grid search over strategy parameters and view results.
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from sandtable.api import run_parameter_sweep
from sandtable.data_types.metric import Metric
from sandtable.ui.components import (
    build_data_handler,
    data_source_selector,
    date_range_picker,
    execution_config_sidebar,
    get_result_store,
    param_grid_builder,
    risk_config_sidebar,
    strategy_selector,
)

st.header("Parameter sweep")
st.caption("Grid search. Exhaustively evaluates all combinations of the parameter values input below.")

#  sidebar
with st.sidebar:
    st.subheader("Universe")
    provider, symbols = data_source_selector(key_prefix="sweep_")

    st.subheader("Date range")
    start_date, end_date = date_range_picker(key_prefix="sweep_")

    initial_capital = st.number_input("Initial capital ($)", value=100_000, step=10_000, key="sweep_capital")
    position_size_pct = st.number_input("Position size (%)", value=10.0, step=1.0, key="sweep_pos_pct") / 100

    optimization_metric = st.selectbox(
        "Optimization metric",
        [m.value for m in Metric],
        index=2,  # sharpe_ratio
        key="sweep_metric",
    )

    exec_config = execution_config_sidebar(key_prefix="sweep_")
    risk_manager = risk_config_sidebar(key_prefix="sweep_")

#  strategy configuration
st.subheader("Strategy")
strategy_cls, _ = strategy_selector(key_prefix="sweep_")

#  parameter grid builder
st.subheader("Parameter grid")
st.caption("Edit the comma-separated values for each parameter to define the search space.")
param_grid = param_grid_builder(strategy_cls, key_prefix="sweep_")

#  run
if st.button("Run sweep", type="primary"):
    if not symbols:
        st.error("Select at least one symbol.")
    elif not param_grid:
        st.error("Define at least one parameter with values.")
    else:
        data = build_data_handler(symbols, start_date, end_date, provider=provider)
        progress_bar = st.progress(0.0, text="Running sweep...")

        total = 1
        for v in param_grid.values():
            total *= len(v)

        try:
            sweep_result = run_parameter_sweep(
                strategy_class=strategy_cls,
                param_grid=param_grid,
                data=data,
                metric=Metric(optimization_metric),
                initial_capital=float(initial_capital),
                position_size_pct=position_size_pct,
                commission=exec_config,
                risk_manager=risk_manager,
                result_store=get_result_store(),
            )
            progress_bar.progress(1.0, text="Sweep complete!")
            st.session_state["sweep_result"] = sweep_result
        except Exception as e:
            st.error(f"Sweep failed: {e}")

#  display results
if "sweep_result" in st.session_state:
    sweep_result = st.session_state["sweep_result"]
    df = sweep_result.to_dataframe()

    st.subheader("Results")
    st.dataframe(df, width="stretch")

    st.success(f"Best params: {sweep_result.best_params}")

    # heatmap for 2-param grids
    param_names = list(param_grid.keys()) if param_grid else []
    if len(param_names) == 2:
        st.subheader("Heatmap")
        try:
            heatmap_df = sweep_result.heatmap_data(param_names[0], param_names[1])
            fig = px.imshow(
                heatmap_df,
                labels=dict(x=param_names[1], y=param_names[0], color=sweep_result.metric),
                title=f"{sweep_result.metric} Heatmap",
                color_continuous_scale="RdYlGn",
                aspect="auto",
            )
            st.plotly_chart(fig, width="stretch")
        except Exception as e:
            st.warning(f"Could not generate heatmap: {e}")
