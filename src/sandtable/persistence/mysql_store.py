"""
src/sandtable/persistence/mysql_store.py

MySQL-backed implementation of AbstractSQLResultStore.

Requires mysql-connector-python: install with `pip install sandtable[databases]`.
"""

from __future__ import annotations

from typing import Any

import mysql.connector

from sandtable.persistence.abstract_sql_result_store import AbstractSQLResultStore
from sandtable.utils.logger import get_logger

logger = get_logger(__name__)

_CREATE_RUNS = """\
CREATE TABLE IF NOT EXISTS runs (
    run_id         VARCHAR(64) PRIMARY KEY,
    strategy_name  VARCHAR(128) NOT NULL,
    config_json    JSON NOT NULL,
    universe       JSON NOT NULL,
    start_date     VARCHAR(64),
    end_date       VARCHAR(64),
    initial_capital DOUBLE NOT NULL,
    sharpe_ratio   DOUBLE,
    sortino_ratio  DOUBLE,
    cagr           DOUBLE,
    max_drawdown   DOUBLE,
    total_return   DOUBLE,
    total_trades   INT,
    win_rate       DOUBLE,
    profit_factor  DOUBLE,
    created_at     VARCHAR(64) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

_CREATE_EQUITY_CURVES = """\
CREATE TABLE IF NOT EXISTS equity_curves (
    run_id    VARCHAR(64) NOT NULL,
    timestamp VARCHAR(64) NOT NULL,
    equity    DOUBLE NOT NULL,
    cash      DOUBLE NOT NULL,
    positions_value DOUBLE NOT NULL,
    PRIMARY KEY (run_id, timestamp),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

_CREATE_FILLS = """\
CREATE TABLE IF NOT EXISTS fills (
    run_id      VARCHAR(64) NOT NULL,
    fill_index  INT NOT NULL,
    symbol      VARCHAR(32) NOT NULL,
    timestamp   VARCHAR(64) NOT NULL,
    direction   VARCHAR(16) NOT NULL,
    quantity    INT NOT NULL,
    fill_price  DOUBLE NOT NULL,
    commission  DOUBLE NOT NULL,
    slippage    DOUBLE NOT NULL,
    market_impact DOUBLE NOT NULL,
    PRIMARY KEY (run_id, fill_index),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

_CREATE_RISK_BREACHES = """\
CREATE TABLE IF NOT EXISTS risk_breaches (
    run_id       VARCHAR(64) NOT NULL,
    breach_index INT NOT NULL,
    timestamp    VARCHAR(64) NOT NULL,
    rule_name    VARCHAR(128) NOT NULL,
    symbol       VARCHAR(32) NOT NULL,
    proposed_qty INT NOT NULL,
    action       VARCHAR(32) NOT NULL,
    breach_value DOUBLE NOT NULL,
    threshold    DOUBLE NOT NULL,
    final_qty    INT,
    PRIMARY KEY (run_id, breach_index),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

_CREATE_TAGS = """\
CREATE TABLE IF NOT EXISTS tags (
    run_id VARCHAR(64) NOT NULL,
    `key`  VARCHAR(128) NOT NULL,
    value  VARCHAR(512) NOT NULL,
    PRIMARY KEY (run_id, `key`),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

_CREATE_INDEXES = [
    "CREATE INDEX idx_runs_strategy ON runs(strategy_name);",
    "CREATE INDEX idx_runs_created ON runs(created_at);",
    "CREATE INDEX idx_runs_sharpe ON runs(sharpe_ratio);",
]


class MySQLResultStore(AbstractSQLResultStore):
    """
    MySQL-backed persistence for backtest results.

    Drop-in replacement for SQLiteResultStore. Requires a running MySQL 8.0+
    instance. Default credentials match the docker-compose.yml in the repo root.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 3306,
        user: str = "sandtable",
        password: str = "sandtable",
        database: str = "sandtable",
    ) -> None:
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._database = database
        self._ensure_schema()
        logger.debug(
            "MySQLResultStore initialized: %s@%s:%d/%s",
            user, host, port, database,
        )

    ## dialect hooks

    @property
    def _placeholder(self) -> str:
        return "%s"

    @property
    def _key_col(self) -> str:
        return "`key`"

    def _connect(self) -> mysql.connector.MySQLConnection:
        """Open a new MySQL connection."""
        logger.debug("Opening MySQL connection to %s:%d", self._host, self._port)
        return mysql.connector.connect(
            host=self._host,
            port=self._port,
            user=self._user,
            password=self._password,
            database=self._database,
        )

    def _dict_cursor(self, conn: Any) -> Any:
        """Return a cursor that returns rows as dicts."""
        return conn.cursor(dictionary=True)

    def _ensure_schema(self) -> None:
        """Create tables and indexes if they don't exist. Idempotent."""
        logger.debug("Ensuring MySQL schema exists")
        conn = self._connect()
        try:
            cur = conn.cursor()
            for ddl in [_CREATE_RUNS, _CREATE_EQUITY_CURVES, _CREATE_FILLS, _CREATE_RISK_BREACHES, _CREATE_TAGS]:
                cur.execute(ddl)
            for idx_sql in _CREATE_INDEXES:
                try:
                    cur.execute(idx_sql)
                except mysql.connector.errors.ProgrammingError:
                    # index already exists
                    pass
            conn.commit()
            logger.debug("MySQL schema ready")
        finally:
            cur.close()
            conn.close()
