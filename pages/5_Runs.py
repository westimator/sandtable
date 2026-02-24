"""
Page 5: Run Browser

Browse, filter, and inspect persisted backtest runs from SQLite.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from sandtable.data_types import ResultBackend
from sandtable.reporting import generate_pdf_tearsheet
from sandtable.ui.components import equity_curve_chart, get_result_store, metrics_table

st.header("Run browser")
st.caption("Browse saved backtest results. Filter by strategy or Sharpe, load full details, or export a PDF tearsheet.")

#  database connection
try:
    store = get_result_store()
except Exception as e:
    st.error(f"Could not connect to database: {e}")
    st.stop()

#  filters
with st.sidebar:
    st.subheader("Filters")
    filter_strategy = st.text_input("Strategy name", key="runs_filter_strategy") or None
    filter_min_sharpe = st.number_input(
        "Min Sharpe",
        value=-999.0,
        step=0.1,
        key="runs_filter_sharpe",
    )
    effective_min_sharpe = filter_min_sharpe if filter_min_sharpe > -999 else None
    limit = st.number_input("Max results", value=50, step=10, min_value=1, key="runs_limit")

#  list runs
_is_mysql = st.session_state.get("store_backend") == ResultBackend.MYSQL

if _is_mysql:
    col_refresh, col_sync = st.columns([1, 1])
    with col_refresh:
        if st.button("Refresh", type="primary"):
            st.session_state.pop("runs_list", None)
    with col_sync:
        if st.button("Sync stores"):
            try:
                from sandtable.persistence import SQLiteResultStore, sync_stores
                from sandtable.persistence.mysql_store import MySQLResultStore

                _sqlite = SQLiteResultStore(db_path=st.session_state.get("db_path", "sandtable.db"))
                _mysql = MySQLResultStore(
                    host=st.session_state.get("mysql_host", "localhost"),
                    port=st.session_state.get("mysql_port", 3306),
                    user=st.session_state.get("mysql_user", "sandtable"),
                    password=st.session_state.get("mysql_password", "sandtable"),
                    database=st.session_state.get("mysql_database", "sandtable"),
                )
                sr = sync_stores(_sqlite, _mysql)
                st.success(
                    f"Sync complete: {sr.copied_a_to_b} SQLite->MySQL, "
                    f"{sr.copied_b_to_a} MySQL->SQLite"
                )
                if sr.total_failed:
                    st.warning(f"{sr.total_failed} runs failed to sync.")
                st.session_state.pop("runs_list", None)
            except Exception as e:
                st.error(f"Sync failed: {e}")
else:
    if st.button("Refresh", type="primary"):
        st.session_state.pop("runs_list", None)

runs = store.list_runs(
    strategy=filter_strategy,
    min_sharpe=effective_min_sharpe,
    limit=int(limit),
)

if not runs:
    st.info("No runs found. Run some backtests first, or check your database path.")
    st.stop()

# display run table
run_data = []
for r in runs:
    run_data.append({
        "Run ID": r.run_id[:8] + "...",
        "Strategy": r.strategy_name,
        "Universe": ", ".join(r.universe),
        "Period": f"{r.start_date[:10]} - {r.end_date[:10]}" if r.start_date else "N/A",
        "Sharpe": f"{r.sharpe_ratio:.2f}",
        "CAGR": f"{r.cagr:.2%}",
        "Max DD": f"{r.max_drawdown:.2%}",
        "Trades": r.total_trades,
        "Created": r.created_at[:19],
        "_full_id": r.run_id,
    })

df = pd.DataFrame(run_data)
display_df = df.drop(columns=["_full_id"])
st.dataframe(display_df, width="stretch", hide_index=True)

#  inspect a run
st.subheader("Inspect run")
st.caption("Select a run to view its equity curve, metrics, trade log, and config.")
run_ids = {f"{r.run_id[:8]}... ({r.strategy_name})": r.run_id for r in runs}
selected_label = st.selectbox("Select run", list(run_ids.keys()), key="runs_select")

if selected_label:
    selected_id = run_ids[selected_label]

    if st.button("Load run details"):
        try:
            config, result = store.load_run(selected_id)
            st.session_state["runs_detail_result"] = result
            st.session_state["runs_detail_config"] = config
        except Exception as e:
            st.error(f"Failed to load run: {e}")

    if "runs_detail_result" in st.session_state:
        result = st.session_state["runs_detail_result"]
        config = st.session_state["runs_detail_config"]

        equity_curve_chart(result, title="Equity Curve")
        metrics_table(result)

        with st.expander("Trade log"):
            trades_df = result.trades_dataframe()
            if not trades_df.empty:
                st.dataframe(trades_df, width="stretch")
            else:
                st.info("No trades.")

        with st.expander("Config JSON"):
            st.json(config.to_dict())

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Regenerate PDF"):
                try:
                    path = generate_pdf_tearsheet(result)
                    st.success(f"PDF saved to: {path}")
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")
        with col2:
            if st.button("Delete run", type="secondary"):
                if store.delete_run(selected_id):
                    st.success("Run deleted.")
                    st.session_state.pop("runs_detail_result", None)
                    st.session_state.pop("runs_detail_config", None)
                    st.rerun()
                else:
                    st.error("Run not found.")
