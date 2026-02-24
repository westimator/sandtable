"""
src/sandtable/ui/components.py

Shared UI components used across dashboard pages.
All charts use plotly for interactivity.
"""

from __future__ import annotations

import inspect
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sandtable.config import BacktestConfig
from sandtable.core.result import BacktestResult
from sandtable.data_engine import CachingProvider, CSVProvider, DataHandler, YFinanceProvider
from sandtable.data_types import DataSource, ResultBackend
from sandtable.execution.simulator import ExecutionConfig
from sandtable.persistence import AbstractResultStore, SQLiteResultStore
from sandtable.persistence.mysql_store import MySQLResultStore
from sandtable.risk import (
    MaxConcentrationRule,
    MaxDailyLossRule,
    MaxDrawdownRule,
    MaxLeverageRule,
    MaxOrderSizeRule,
    MaxPortfolioExposureRule,
    MaxPositionSizeRule,
    RiskManager,
)
from sandtable.strategy.abstract_strategy import AbstractStrategy
from sandtable.strategy.buy_and_hold_strategy import BuyAndHoldStrategy
from sandtable.strategy.ma_crossover_strategy import MACrossoverStrategy
from sandtable.strategy.mean_reversion_strategy import MeanReversionStrategy

# registry of available strategies
STRATEGY_REGISTRY: dict[str, type[AbstractStrategy]] = {
    "Buy & Hold": BuyAndHoldStrategy,
    "MA Crossover": MACrossoverStrategy,
    "Mean Reversion": MeanReversionStrategy,
}

# bundled CSV fixture symbols
CSV_FIXTURE_SYMBOLS: list[str] = ["SPY", "QQQ", "IWM", "AAPL", "MSFT"]

_DATA_SOURCE_LABELS: dict[DataSource, str] = {
    DataSource.CSV: "CSV (bundled)",
    DataSource.YFINANCE: "Yahoo Finance",
}

_LABEL_TO_SOURCE: dict[str, DataSource] = {v: k for k, v in _DATA_SOURCE_LABELS.items()}


def get_result_store() -> AbstractResultStore:
    """
    Return a cached result store based on the sidebar backend selection.

    Supports SQLite (default) and MySQL.
    """
    backend = st.session_state.get("store_backend", ResultBackend.SQLITE)
    if backend == ResultBackend.MYSQL:
        cache_key = "_result_store_mysql"
        if cache_key not in st.session_state:
            st.session_state[cache_key] = MySQLResultStore(
                host=st.session_state.get("mysql_host", "localhost"),
                port=st.session_state.get("mysql_port", 3306),
                user=st.session_state.get("mysql_user", "sandtable"),
                password=st.session_state.get("mysql_password", "sandtable"),
                database=st.session_state.get("mysql_database", "sandtable"),
            )
        return st.session_state[cache_key]
    elif backend == ResultBackend.SQLITE:
        db_path = st.session_state.get("db_path", "sandtable.db")
        cache_key = f"_result_store_{db_path}"
        if cache_key not in st.session_state:
            st.session_state[cache_key] = SQLiteResultStore(db_path=db_path)
        return st.session_state[cache_key]
    else:
        raise ValueError(f"Unknown result backend: {backend}")


def strategy_selector(key_prefix: str = "") -> tuple[type[AbstractStrategy], dict[str, Any]]:
    """
    Strategy dropdown + dynamic parameter inputs.

    Returns:
        (strategy_class, params_dict)
    """
    name = st.selectbox(
        "Strategy",
        list(STRATEGY_REGISTRY.keys()),
        key=f"{key_prefix}strategy_select",
    )
    cls = STRATEGY_REGISTRY[name]

    # inspect constructor to build dynamic inputs
    sig = inspect.signature(obj=cls.__init__)
    params: dict[str, Any] = {}
    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        default = param.default if param.default is not inspect.Parameter.empty else 0
        if isinstance(default, int):
            params[pname] = st.number_input(
                pname.replace("_", " ").title(),
                value=default,
                step=1,
                key=f"{key_prefix}param_{pname}",
            )
        elif isinstance(default, float):
            params[pname] = st.number_input(
                pname.replace("_", " ").title(),
                value=default,
                step=0.1,
                format="%.2f",
                key=f"{key_prefix}param_{pname}",
            )

    return cls, params,


def execution_config_sidebar(key_prefix: str = "") -> ExecutionConfig:
    """Sidebar inputs for execution costs. Returns ExecutionConfig."""
    with st.expander("Execution Config"):
        commission_per_share = st.number_input(
            "Commission per share ($)",
            value=0.005,
            step=0.001,
            format="%.3f",
            key=f"{key_prefix}comm_per_share",
        )
        commission_pct = st.number_input(
            "Commission (%)",
            value=0.0,
            step=0.001,
            format="%.4f",
            key=f"{key_prefix}comm_pct",
        )
        commission_minimum = st.number_input(
            "Minimum commission ($)",
            value=1.0,
            step=0.5,
            key=f"{key_prefix}comm_min",
        )
    return ExecutionConfig(
        commission_per_share=commission_per_share,
        commission_pct=commission_pct,
        commission_minimum=commission_minimum,
    )


def risk_config_sidebar(key_prefix: str = "") -> RiskManager | None:
    """Toggle switches for risk rules. Returns RiskManager or None."""
    with st.expander("Risk Rules"):
        rules = []

        if st.checkbox("Max Position Size", key=f"{key_prefix}risk_pos"):
            val = st.number_input("Max position (%)", value=20.0, step=1.0, key=f"{key_prefix}risk_pos_val") / 100
            rules.append(MaxPositionSizeRule(max_position_pct=val))

        if st.checkbox("Max Portfolio Exposure", key=f"{key_prefix}risk_exp"):
            val = st.number_input("Max exposure (%)", value=100.0, step=5.0, key=f"{key_prefix}risk_exp_val") / 100
            rules.append(MaxPortfolioExposureRule(max_gross_exposure_pct=val))

        if st.checkbox("Max Drawdown", key=f"{key_prefix}risk_dd"):
            val = st.number_input("Max drawdown (%)", value=20.0, step=1.0, key=f"{key_prefix}risk_dd_val") / 100
            rules.append(MaxDrawdownRule(max_drawdown_pct=val))

        if st.checkbox("Max Daily Loss", key=f"{key_prefix}risk_daily"):
            val = st.number_input("Max daily loss (%)", value=5.0, step=0.5, key=f"{key_prefix}risk_daily_val") / 100
            rules.append(MaxDailyLossRule(max_daily_loss_pct=val))

        if st.checkbox("Max Leverage", key=f"{key_prefix}risk_lev"):
            val = st.number_input("Max leverage", value=2.0, step=0.1, key=f"{key_prefix}risk_lev_val")
            rules.append(MaxLeverageRule(max_leverage=val))

        if st.checkbox("Max Order Size", key=f"{key_prefix}risk_ord"):
            val = st.number_input("Max order (shares)", value=10000, step=100, key=f"{key_prefix}risk_ord_val")
            rules.append(MaxOrderSizeRule(max_order_qty=val))

        if st.checkbox("Max Concentration", key=f"{key_prefix}risk_conc"):
            val = st.number_input("Max concentration (%)", value=30.0, step=1.0, key=f"{key_prefix}risk_conc_val") / 100
            rules.append(MaxConcentrationRule(max_concentration_pct=val))

    if not rules:
        return None
    return RiskManager(rules=rules)


def data_source_selector(key_prefix: str = "") -> tuple[object, list[str]]:
    """
    Data source selection with symbol inputs.

    The data *source* (CSV vs Yahoo Finance) is a per-run choice shown
    as a sidebar widget.

    Returns:
        (provider, symbols)
    """
    # source selector (per-run choice)
    source_label = st.selectbox(
        "Data source",
        list(_DATA_SOURCE_LABELS.values()),
        key=f"{key_prefix}data_source",
    )
    source = _LABEL_TO_SOURCE[source_label]

    # symbol selection follows the source
    if source == DataSource.CSV:
        symbols = st.multiselect(
            "Symbols",
            CSV_FIXTURE_SYMBOLS,
            default=["SPY"],
            key=f"{key_prefix}symbols",
        )
    elif source == DataSource.YFINANCE:
        ticker_input = st.text_input(
            "Tickers (comma-separated)",
            value="SPY",
            key=f"{key_prefix}yf_tickers",
        )
        symbols = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    else:
        raise ValueError(f"Unknown data source: {source}")

    if source == DataSource.CSV:
        provider = CSVProvider(data_dir="data/fixtures")
    else:
        provider = CachingProvider(provider=YFinanceProvider())

    return provider, symbols


def date_range_picker(key_prefix: str = "") -> tuple[str, str]:
    """Start/end date inputs. Returns (start_str, end_str)."""
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input(
            "Start date",
            value=pd.Timestamp("2018-01-01"),
            key=f"{key_prefix}start_date",
        )
    with col2:
        end = st.date_input(
            "End date",
            value=pd.Timestamp("2023-12-31"),
            key=f"{key_prefix}end_date",
        )
    return str(start), str(end)


def metrics_table(result: BacktestResult) -> None:
    """Display key metrics in a formatted table."""
    m = result.metrics
    data = {
        "Metric": [
            "Total Return",
            "CAGR",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Max Drawdown",
            "Win Rate",
            "Profit Factor",
            "Avg Trade PnL",
            "Num Trades",
        ],
        "Value": [
            f"{m.total_return:.2%}",
            f"{m.cagr:.2%}",
            f"{m.sharpe_ratio:.2f}",
            f"{m.sortino_ratio:.2f}",
            f"{m.max_drawdown:.2%}",
            f"{m.win_rate:.2%}",
            f"{m.profit_factor:.2f}",
            f"${m.avg_trade_pnl:,.2f}",
            str(m.num_trades),
        ],
    }
    st.dataframe(
        data=pd.DataFrame(data),
        hide_index=True,
        width="stretch",
    )


def equity_curve_chart(result: BacktestResult, title: str = "Equity Curve") -> None:
    """Plotly equity curve with drawdown overlay."""
    eq_df = result.equity_dataframe()
    if eq_df.empty:
        st.warning("No equity data to plot.")
        return

    eq_df["peak"] = eq_df["equity"].cummax()
    eq_df["drawdown"] = (eq_df["equity"] - eq_df["peak"]) / eq_df["peak"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq_df["timestamp"],
        y=eq_df["equity"],
        name="Equity",
        line=dict(color="#2196F3"),
    ))
    fig.add_trace(go.Scatter(
        x=eq_df["timestamp"],
        y=eq_df["drawdown"],
        name="Drawdown",
        yaxis="y2",
        fill="tozeroy",
        line=dict(color="#F44336"),
        opacity=0.3,
    ))
    fig.update_layout(
        title=title,
        yaxis=dict(title="Equity ($)", side="left"),
        yaxis2=dict(title="Drawdown", side="right", overlaying="y", tickformat=".0%", range=[-0.5, 0]),
        hovermode="x unified",
        height=450,
    )
    st.plotly_chart(fig, width="stretch")


def equity_curves_overlay(
    equity_data: dict[str, pd.DataFrame],
    title: str = "Equity Curves",
    *,
    dash_names: set[str] | None = None,
) -> None:
    """Overlay multiple equity curves on one plotly chart."""
    dash_names = dash_names or set()
    fig = go.Figure()
    for name, eq_df in equity_data.items():
        line_style = dict(dash="dash") if name in dash_names else {}
        fig.add_trace(go.Scatter(
            x=eq_df["timestamp"],
            y=eq_df["equity"],
            name=name,
            line=line_style,
        ))
    fig.update_layout(
        title=title,
        yaxis=dict(title="Equity ($)"),
        hovermode="x unified",
        height=450,
    )
    st.plotly_chart(fig, width="stretch")


def config_expander(config: BacktestConfig) -> None:
    """Show BacktestConfig JSON in an expander."""
    with st.expander("Full BacktestConfig JSON"):
        st.json(config.to_dict())


def config_dict_expander(config_dict: dict[str, Any]) -> None:
    """Show an arbitrary config dict in an expander."""
    with st.expander("Configuration"):
        st.json(config_dict)


def _default_int_range(default: int) -> list[int]:
    """
    Generate ~4 sensible integer sweep values centered tightly around a default.

    Keeps the spread within roughly +/-50% of the default so that parameters
    with different scales (e.g. fast_period=10 vs slow_period=30) don't
    produce overlapping ranges.
    """
    if default <= 3:
        return list(range(1, default + 3))
    step = max(1, default // 4)
    values = [default - step, default, default + step, default + 2 * step]
    return [v for v in values if v > 0]


def _default_float_range(default: float) -> list[float]:
    """
    Generate ~4 sensible float sweep values centered tightly around a default.
    """
    step = abs(default) / 4 if default != 0.0 else 0.5
    values = [default - step, default, default + step, default + 2 * step]
    return [round(v, 4) for v in values if v > 0]


def param_grid_builder(strategy_cls: type[AbstractStrategy], key_prefix: str = "") -> dict[str, list]:
    """
    Auto-detect strategy parameters and render a grid builder UI.

    Inspects the strategy __init__ to find tunable parameters (excluding
    self and max_history), then renders one row per parameter with a
    checkbox and comma-separated value input pre-filled with defaults.

    Returns:
        param_grid dict suitable for run_parameter_sweep / run_walkforward.
    """
    sig = inspect.signature(obj=strategy_cls.__init__)
    tunable = []
    for pname, param in sig.parameters.items():
        if pname in ("self", "max_history"):
            continue
        tunable.append((pname, param))

    if not tunable:
        st.info("This strategy has no tunable parameters.")
        return {}

    param_grid: dict[str, list] = {}
    for pname, param in tunable:
        default = param.default if param.default is not inspect.Parameter.empty else 0
        # generate default range string
        if isinstance(default, float):
            default_values = _default_float_range(default)
            default_str = ", ".join(str(v) for v in default_values)
        else:
            default_values = _default_int_range(int(default))
            default_str = ", ".join(str(v) for v in default_values)

        col_check, col_label, col_vals = st.columns([0.5, 1.5, 3])
        with col_check:
            include = st.checkbox("Include", value=True, key=f"{key_prefix}grid_inc_{pname}", label_visibility="collapsed")
        with col_label:
            st.markdown(f"**{pname.replace('_', ' ').title()}**")
        with col_vals:
            pvals = st.text_input(
                f"Values for {pname}",
                value=default_str,
                key=f"{key_prefix}grid_vals_{pname}",
                label_visibility="collapsed",
            )

        if include and pvals:
            try:
                values = [int(v.strip()) if v.strip().lstrip("-").isdigit() else float(v.strip()) for v in pvals.split(",")]
                param_grid[pname] = values
            except ValueError:
                st.warning(f"Could not parse values for {pname}")

    if param_grid:
        total = 1
        for v in param_grid.values():
            total *= len(v)
        st.caption(f"{total} combinations")

    return param_grid


def build_data_handler(symbols: list[str], start_date: str, end_date: str, provider=None):
    """
    Build a DataHandler with date slicing.

    Defaults to CSVProvider with bundled fixtures when provider is None.
    """
    if provider is None:
        provider = CSVProvider(data_dir="data/fixtures")

    handler = DataHandler(
        provider=provider,
        universe=symbols,
    )
    handler.load(
        start_date=start_date,
        end_date=end_date,
    )
    return handler
