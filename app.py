"""
app.py

Streamlit entry point for Sandtable dashboard.

Backend configuration (result store) is set once at startup via CLI flags
and locked for the session. Per-run settings (strategy, symbols, dates,
execution, risk) remain in the sidebar.

Run with:
    # default: SQLite result store, in-memory data from bundled CSVs
    uv run streamlit run app.py

    # use MySQL for result persistence (requires docker compose up -d)
    uv run streamlit run app.py -- --result-backend mysql
"""

import argparse
import sys

import streamlit as st

from sandtable.data_types import ResultBackend
from sandtable.utils.logger import get_logger

logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse CLI args passed after ``--`` in ``streamlit run app.py -- ...``."""
    parser = argparse.ArgumentParser(description="Sandtable dashboard")

    # backends
    parser.add_argument(
        "--result-backend",
        choices=["sqlite", "mysql"],
        default="sqlite",
        help="Where to persist backtest results (default: sqlite)",
    )
    # SQLite
    parser.add_argument("--db-path", default="sandtable.db", help="SQLite database path")

    # MySQL
    parser.add_argument("--mysql-host", default="localhost")
    parser.add_argument("--mysql-port", type=int, default=3306)
    parser.add_argument("--mysql-user", default="sandtable")
    parser.add_argument("--mysql-password", default="sandtable")
    parser.add_argument("--mysql-database", default="sandtable")

    return parser.parse_args(sys.argv[1:])


st.set_page_config(
    page_title="Sandtable",
    layout="wide",
    page_icon="\u2693",
)

# parse CLI args and lock backend config into session state (once per session)
if "_backends_configured" not in st.session_state:
    args = _parse_args()

    st.session_state["store_backend"] = ResultBackend(args.result_backend)

    # SQLite
    st.session_state["db_path"] = args.db_path

    # MySQL
    st.session_state["mysql_host"] = args.mysql_host
    st.session_state["mysql_port"] = args.mysql_port
    st.session_state["mysql_user"] = args.mysql_user
    st.session_state["mysql_password"] = args.mysql_password
    st.session_state["mysql_database"] = args.mysql_database

    st.session_state["_backends_configured"] = True

# auto-sync SQLite <-> MySQL on first load when MySQL is selected
store_backend = st.session_state["store_backend"]
if store_backend == ResultBackend.MYSQL and not st.session_state.get("_sync_done"):
    try:
        from sandtable.persistence import SQLiteResultStore, sync_stores
        from sandtable.persistence.mysql_store import MySQLResultStore

        _sqlite = SQLiteResultStore(
            db_path=st.session_state.get("db_path", "sandtable.db"),
        )
        _mysql = MySQLResultStore(
            host=st.session_state.get("mysql_host", "localhost"),
            port=st.session_state.get("mysql_port", 3306),
            user=st.session_state.get("mysql_user", "sandtable"),
            password=st.session_state.get("mysql_password", "sandtable"),
            database=st.session_state.get("mysql_database", "sandtable"),
        )
        _sr = sync_stores(_sqlite, _mysql)
        st.session_state["_sync_done"] = True
        if _sr.total_copied > 0:
            logger.info("Startup sync: %d runs copied", _sr.total_copied)
    except Exception:
        # silently skip if MySQL is unreachable
        logger.debug("Startup sync skipped, MySQL unreachable", exc_info=True)

# home page
st.title("Sandtable")
st.markdown("Event-driven backtesting framework with realistic execution modeling.")
st.markdown(
    "Use the sidebar to navigate between pages: **Backtest**, **Sweep**, "
    "**Walk-Forward**, **Compare**, and **Run Browser**."
)

# show active backend config (read-only)
st.subheader("Active configuration")

st.markdown("**Result store**")
if store_backend == ResultBackend.SQLITE:
    st.code(f"SQLite - {st.session_state['db_path']}", language=None)
else:
    st.code(
        f"MySQL - {st.session_state['mysql_user']}@"
        f"{st.session_state['mysql_host']}:{st.session_state['mysql_port']}/"
        f"{st.session_state['mysql_database']}",
        language=None,
    )

st.caption(
    "Backend configuration is set at startup via CLI flags. "
    "Restart the app with different flags to change backends. "
    "Run ``uv run streamlit run app.py -- --help`` for options."
)
