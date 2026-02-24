"""
Page 1: Single Backtest

Run a single backtest with configurable strategy, execution, and risk parameters.
"""

from __future__ import annotations

import streamlit as st

from sandtable.api import run_backtest
from sandtable.config import BacktestConfig
from sandtable.reporting import compute_tca, generate_pdf_tearsheet
from sandtable.ui.components import (
    build_data_handler,
    config_expander,
    data_source_selector,
    date_range_picker,
    equity_curve_chart,
    execution_config_sidebar,
    get_result_store,
    metrics_table,
    risk_config_sidebar,
    strategy_selector,
)

st.header("Single backtest")
st.caption("Run one strategy against historical data and review equity curve, metrics, trade log, and cost breakdown.")

# sidebar
with st.sidebar:
    st.subheader("Universe")
    provider, symbols = data_source_selector(key_prefix="bt_")

    st.subheader("Date range")
    start_date, end_date = date_range_picker()

    initial_capital = st.number_input("Initial capital ($)", value=100_000, step=10_000, key="bt_capital")
    position_size_pct = st.number_input("Position size (%)", value=10.0, step=1.0, key="bt_pos_pct") / 100

    exec_config = execution_config_sidebar()
    risk_manager = risk_config_sidebar()

# strategy configuration
st.subheader("Strategy")
st.caption("Pick a strategy and adjust its parameters. These are the exact values used for this run.")
strategy_cls, strategy_params = strategy_selector()

# main area
if st.button("Run backtest", type="primary"):
    if not symbols:
        st.error("Select at least one symbol.")
    else:
        with st.spinner("Running backtest..."):
            try:
                data = build_data_handler(symbols, start_date, end_date, provider=provider)
                strategy = strategy_cls(**strategy_params)
                result = run_backtest(
                    strategy=strategy,
                    data=data,
                    initial_capital=float(initial_capital),
                    position_size_pct=position_size_pct,
                    commission=exec_config,
                    risk_manager=risk_manager,
                    result_store=get_result_store(),
                )
                st.session_state["bt_result"] = result
                st.session_state["bt_config"] = BacktestConfig(
                    strategy_cls=strategy_cls,
                    strategy_params=strategy_params,
                    universe=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=float(initial_capital),
                    position_size_pct=position_size_pct,
                )
            except Exception as e:
                st.error(f"Backtest failed: {e}")

# display results
if "bt_result" in st.session_state:
    result = st.session_state["bt_result"]

    equity_curve_chart(result)

    st.subheader("Key metrics")
    metrics_table(result)

    # trade log
    with st.expander("Trade log"):
        trades_df = result.trades_dataframe()
        if not trades_df.empty:
            st.dataframe(trades_df, width="stretch")
        else:
            st.info("No trades executed.")

    # TCA breakdown
    with st.expander("TCA breakdown"):
        if result.trades:
            tca = compute_tca(result.trades, result.metrics.total_return * result.initial_capital)
            st.metric("Total slippage", f"${tca.total_slippage:,.2f}")
            st.metric("Total market impact", f"${tca.total_impact:,.2f}")
            st.metric("Total commission", f"${tca.total_commission:,.2f}")
            st.metric("Total cost", f"${tca.total_cost:,.2f}")
        else:
            st.info("No trades to analyze.")

    # config
    if "bt_config" in st.session_state:
        config_expander(st.session_state["bt_config"])

    # PDF generation
    if st.button("Generate PDF tearsheet"):
        with st.spinner("Generating PDF..."):
            try:
                path = generate_pdf_tearsheet(result)
                st.success(f"PDF saved to: {path}")
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
