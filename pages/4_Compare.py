"""
Page 4: Strategy Comparison

Compare 2-4 strategies side by side with overlaid equity curves,
metrics table, and correlation matrix.
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from sandtable.api import run_backtest
from sandtable.reporting import generate_comparison_report
from sandtable.research import run_comparison
from sandtable.ui.components import (
    build_data_handler,
    data_source_selector,
    date_range_picker,
    equity_curves_overlay,
    execution_config_sidebar,
    get_result_store,
    risk_config_sidebar,
    strategy_selector,
)

st.header("Strategy comparison")
st.caption("Run 2-4 strategies on the same data and compare equity curves, metrics, and return correlations side by side.")

#  sidebar (shared settings)
with st.sidebar:
    st.subheader("Shared settings")
    provider, symbols = data_source_selector(key_prefix="cmp_")
    start_date, end_date = date_range_picker(key_prefix="cmp_")
    initial_capital = st.number_input("Initial capital ($)", value=100_000, step=10_000, key="cmp_capital")
    position_size_pct = st.number_input("Position size (%)", value=10.0, step=1.0, key="cmp_pos_pct") / 100
    exec_config = execution_config_sidebar(key_prefix="cmp_")
    risk_manager = risk_config_sidebar(key_prefix="cmp_")

#  strategy configuration
num_strategies = st.number_input("Number of strategies", min_value=2, max_value=4, value=2, key="cmp_num")

strategy_configs: list[tuple[str, type, dict]] = []
for i in range(int(num_strategies)):
    with st.expander(f"Strategy {i+1}", expanded=True):
        cls, params = strategy_selector(key_prefix=f"cmp_{i}_")
        # build label from strategy-specific params (exclude max_history)
        display_params = {k: v for k, v in params.items() if k != "max_history"}
        param_str = ", ".join(f"{k}={v}" for k, v in display_params.items())
        auto_label = f"{cls.__name__}({param_str})" if param_str else cls.__name__
        # auto-update label when strategy or params change
        auto_key = f"_cmp_auto_label_{i}"
        label_key = f"cmp_name_{i}"
        if st.session_state.get(auto_key) != auto_label:
            st.session_state[auto_key] = auto_label
            st.session_state[label_key] = auto_label
        name = st.text_input("Label", key=label_key)
        strategy_configs.append((name, cls, params))

#  run
if st.button("Run comparison", type="primary"):
    if not symbols:
        st.error("Select at least one symbol.")
    elif len(strategy_configs) < 2:
        st.error("Configure at least 2 strategies.")
    else:
        with st.spinner("Running backtests..."):
            try:
                data = build_data_handler(symbols, start_date, end_date, provider=provider)
                store = get_result_store()
                results = {}
                for name, cls, params in strategy_configs:
                    data.reset()
                    strategy = cls(**params)
                    result = run_backtest(
                        strategy=strategy,
                        data=data,
                        initial_capital=float(initial_capital),
                        position_size_pct=position_size_pct,
                        commission=exec_config,
                        risk_manager=risk_manager,
                        result_store=store,
                    )
                    results[name] = result

                comparison = run_comparison(results)
                st.session_state["cmp_result"] = comparison
                st.session_state["cmp_results"] = results
            except Exception as e:
                st.error(f"Comparison failed: {e}")

#  display results
if "cmp_result" in st.session_state:
    comparison = st.session_state["cmp_result"]
    results = st.session_state["cmp_results"]

    # overlaid equity curves (including blended portfolio)
    st.subheader("Equity curves")
    equity_data = {}
    for name, result in results.items():
        equity_data[name] = result.equity_dataframe()
    if comparison.blended_equity_curve:
        import pandas as pd

        equity_data["Blended (equal weight)"] = pd.DataFrame([
            {"timestamp": p.timestamp, "equity": p.equity}
            for p in comparison.blended_equity_curve
        ])
    equity_curves_overlay(equity_data, dash_names={"Blended (equal weight)"})

    # side-by-side metrics
    st.subheader("Performance table")
    st.dataframe(comparison.performance_table, width="stretch")

    # correlation matrix
    st.subheader("Return correlation matrix")
    corr = comparison.correlation_matrix
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, width="stretch")

    # PDF report
    if st.button("Generate comparison PDF"):
        with st.spinner("Generating PDF..."):
            try:
                path = generate_comparison_report(results, correlation_matrix=corr)
                st.success(f"PDF saved to: {path}")
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
