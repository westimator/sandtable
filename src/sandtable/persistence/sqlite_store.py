"""
src/sandtable/persistence/sqlite_store.py

SQLite-backed implementation of AbstractSQLResultStore.

Uses stdlib sqlite3 only - no SQLAlchemy.
"""

from __future__ import annotations

import sqlite3
import textwrap
from typing import Any

from sandtable.persistence.abstract_sql_result_store import AbstractSQLResultStore
from sandtable.utils.logger import get_logger

logger = get_logger(__name__)

_SCHEMA_SQL = textwrap.dedent("""\
    CREATE TABLE IF NOT EXISTS runs (
        run_id         TEXT PRIMARY KEY,
        strategy_name  TEXT NOT NULL,
        config_json    TEXT NOT NULL,
        universe       TEXT NOT NULL,
        start_date     TEXT,
        end_date       TEXT,
        initial_capital REAL NOT NULL,
        sharpe_ratio   REAL,
        sortino_ratio  REAL,
        cagr           REAL,
        max_drawdown   REAL,
        total_return   REAL,
        total_trades   INTEGER,
        win_rate       REAL,
        profit_factor  REAL,
        created_at     TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_runs_strategy ON runs(strategy_name);
    CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at);
    CREATE INDEX IF NOT EXISTS idx_runs_sharpe ON runs(sharpe_ratio);

    CREATE TABLE IF NOT EXISTS equity_curves (
        run_id    TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        equity    REAL NOT NULL,
        cash      REAL NOT NULL,
        positions_value REAL NOT NULL,
        PRIMARY KEY (run_id, timestamp),
        FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS fills (
        run_id      TEXT NOT NULL,
        fill_index  INTEGER NOT NULL,
        symbol      TEXT NOT NULL,
        timestamp   TEXT NOT NULL,
        direction   TEXT NOT NULL,
        quantity    INTEGER NOT NULL,
        fill_price  REAL NOT NULL,
        commission  REAL NOT NULL,
        slippage    REAL NOT NULL,
        market_impact REAL NOT NULL,
        PRIMARY KEY (run_id, fill_index),
        FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS risk_breaches (
        run_id       TEXT NOT NULL,
        breach_index INTEGER NOT NULL,
        timestamp    TEXT NOT NULL,
        rule_name    TEXT NOT NULL,
        symbol       TEXT NOT NULL,
        proposed_qty INTEGER NOT NULL,
        action       TEXT NOT NULL,
        breach_value REAL NOT NULL,
        threshold    REAL NOT NULL,
        final_qty    INTEGER,
        PRIMARY KEY (run_id, breach_index),
        FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS tags (
        run_id TEXT NOT NULL,
        key    TEXT NOT NULL,
        value  TEXT NOT NULL,
        PRIMARY KEY (run_id, key),
        FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
    );
""")


class SQLiteResultStore(AbstractSQLResultStore):
    """
    SQLite-backed persistence for backtest results.

    Creates the database and schema on first use. Uses WAL mode for
    concurrent read/write safety.
    """

    def __init__(self, db_path: str = "sandtable.db") -> None:
        self._db_path = db_path
        self._ensure_schema()
        logger.debug("SQLiteResultStore initialized with db_path=%s", db_path)

    ## dialect hooks

    @property
    def _placeholder(self) -> str:
        return "?"

    @property
    def _key_col(self) -> str:
        return "key"

    def _connect(self) -> sqlite3.Connection:
        """Open a connection with foreign keys enabled."""
        logger.debug("Opening SQLite connection to %s", self._db_path)
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _dict_cursor(self, conn: Any) -> sqlite3.Cursor:
        """Return a cursor whose rows support column-name access."""
        conn.row_factory = sqlite3.Row
        return conn.cursor()

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist. Idempotent."""
        logger.debug("Ensuring schema exists for %s", self._db_path)
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(_SCHEMA_SQL)
            conn.commit()
            logger.debug("schema ready")
        finally:
            conn.close()
