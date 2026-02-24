"""
src/sandtable/data_types/result_backend.py

Where backtest results are persisted.

Orthogonal to DataSource (where market data originates).
"""

from __future__ import annotations

from enum import StrEnum


class ResultBackend(StrEnum):
    """
    Where backtest results are stored.

    Attributes:
        SQLITE: Results persisted to a local SQLite database file (default). Runs as thread in main backtester process.
        MYSQL: Results persisted to a MySQL 8.0+ server. Runs in separate process, see docker-compose.yml.
    """
    SQLITE = "sqlite"
    MYSQL = "mysql"
