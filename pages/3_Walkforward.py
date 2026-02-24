"""
Page 3: Walk-Forward Analysis

Evaluate strategies with rolling train/test folds to detect overfitting.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sandtable.data_types.metric import Metric
from sandtable.research import run_walkforward
from sandtable.ui.components import (
    build_data_handler,
    config_dict_expander,
    data_source_selector,
    date_range_picker,
    execution_config_sidebar,
    get_result_store,
    param_grid_builder,
    risk_config_sidebar,
    strategy_selector,
)

st.header("Walk-forward analysis")
st.caption(
    "Simulates realistic strategy selection by splitting history into rolling train/test windows. "
    "Each fold optimizes parameters on the training set, then evaluates on unseen test data. "
    "If out-of-sample performance holds up, the strategy is less likely to be overfit."
)

#  sidebar
with st.sidebar:
    st.subheader("Universe")
    provider, symbols = data_source_selector(key_prefix="wf_")

    st.subheader("Date range")
    start_date, end_date = date_range_picker(key_prefix="wf_")

    initial_capital = st.number_input("Initial capital ($)", value=100_000, step=10_000, key="wf_capital")
    position_size_pct = st.number_input("Position size (%)", value=10.0, step=1.0, key="wf_pos_pct") / 100

    st.subheader("Walk-forward settings")
    train_window = st.number_input("Train window (days)", value=504, step=21, key="wf_train")
    test_window = st.number_input("Test window (days)", value=126, step=21, key="wf_test")
    step_size = st.number_input("Step size (days, 0 = test window)", value=0, step=21, key="wf_step")

    optimization_metric = st.selectbox(
        "Optimization metric",
        [m.value for m in Metric],
        index=2,
        key="wf_metric",
    )

    exec_config = execution_config_sidebar(key_prefix="wf_")
    risk_manager = risk_config_sidebar(key_prefix="wf_")

#  strategy configuration
st.subheader("Strategy")
strategy_cls, _ = strategy_selector(key_prefix="wf_")

#  parameter grid
st.subheader("Parameter grid")
st.caption("Search space for in-sample optimization. Each fold picks the best combination from these values.")
param_grid = param_grid_builder(strategy_cls, key_prefix="wf_")

#  run
if st.button("Run walk-forward", type="primary"):
    if not symbols:
        st.error("Select at least one symbol.")
    elif not param_grid:
        st.error("Define at least one parameter with values.")
    else:
        data = build_data_handler(symbols, start_date, end_date, provider=provider)
        effective_step = int(step_size) if int(step_size) > 0 else None

        with st.spinner("Running walk-forward analysis..."):
            try:
                wf_result = run_walkforward(
                    strategy_cls=strategy_cls,
                    param_grid=param_grid,
                    data=data,
                    train_window=int(train_window),
                    test_window=int(test_window),
                    step_size=effective_step,
                    optimization_metric=Metric(optimization_metric),
                    initial_capital=float(initial_capital),
                    position_size_pct=position_size_pct,
                    commission=exec_config,
                    risk_manager=risk_manager,
                    result_store=get_result_store(),
                )
                st.session_state["wf_result"] = wf_result
            except Exception as e:
                st.error(f"Walk-forward failed: {e}")

#  display results
if "wf_result" in st.session_state:
    wf_result = st.session_state["wf_result"]

    # summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("OOS Sharpe", f"{wf_result.oos_sharpe:.2f}")
    col2.metric("OOS max drawdown", f"{wf_result.oos_max_drawdown:.2%}")
    col3.metric("OOS CAGR", f"{wf_result.oos_cagr:.2%}")

    # per-fold table
    st.subheader("Per-fold results")
    st.caption("Each row is one train/test window. Compare IS vs OOS Sharpe - a large drop suggests overfitting.")
    fold_rows = []
    for fold in wf_result.folds:
        fold_rows.append({
            "Fold": fold.fold_index,
            "Train": f"{fold.train_start} - {fold.train_end}",
            "Test": f"{fold.test_start} - {fold.test_end}",
            "Chosen params": str(fold.chosen_params),
            "IS Sharpe": fold.in_sample_metrics.get(Metric.SHARPE_RATIO, 0.0),
            "OOS Sharpe": fold.out_of_sample_metrics.get(Metric.SHARPE_RATIO, 0.0),
            "OOS return": fold.out_of_sample_metrics.get(Metric.TOTAL_RETURN, 0.0),
        })
    st.dataframe(pd.DataFrame(fold_rows), width="stretch", hide_index=True)

    # stitched OOS equity curve
    if wf_result.oos_equity_curve:
        st.subheader("Stitched OOS equity curve")
        st.caption("OOS segments joined end-to-end. This is what the strategy would have returned with no lookahead.")
        eq_data = [{"timestamp": p.timestamp, "equity": p.equity} for p in wf_result.oos_equity_curve]
        eq_df = pd.DataFrame(eq_data)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=eq_df["timestamp"],
            y=eq_df["equity"],
            name="OOS equity",
            line=dict(color="#4CAF50"),
        ))
        fig.update_layout(
            yaxis=dict(title="Equity ($)"),
            hovermode="x unified",
            height=400,
        )
        st.plotly_chart(fig, width="stretch")

    config_dict_expander(wf_result.config)
