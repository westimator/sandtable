"""
src/sandtable/data_types/data_source.py

Supported data sources (where market data originates).
"""

from __future__ import annotations

from enum import StrEnum


class DataSource(StrEnum):
    """
    Where market data originates from.

    Attributes:
        CSV: Load bars from local CSV files in data/fixtures/ (bundled SPY, QQQ, IWM, AAPL, MSFT 2018-2023).
        YFINANCE: Download bars on the fly via the yfinance library.
    """
    CSV = "csv"
    YFINANCE = "yfinance"
