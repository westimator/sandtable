"""
src/sandtable/data_handlers/csv_data_handler.py

Data handler for loading and iterating over OHLCV data from CSV files.
"""

from pathlib import Path

import pandas as pd

from sandtable.data_handlers.single_symbol_handler import SingleSymbolDataHandler
from sandtable.utils.logger import get_logger

logger = get_logger(__name__)


class CSVDataHandler(SingleSymbolDataHandler):
    """
    Loads and iterates over OHLCV data from CSV files.

    Enforces temporal causality by only allowing access to historical data
    up to (but not including) the current bar index.

    Attributes:
        symbol: The ticker symbol for this data

    Example:
        >>> handler = CSVDataHandler("data/spy.csv", "SPY")
        >>> while (bar := handler.get_next_bar()) is not None:
        ...     historical = handler.get_historical_bars(20)
        ...     process(bar, historical)
    """

    def __init__(self, filepath: str | Path, symbol: str) -> None:
        """
        Initialize the data handler with a CSV file.

        Args:
            filepath: Path to CSV file with OHLCV data.
                     Expected columns: Date, Open, High, Low, Close, Volume
            symbol: Ticker symbol for this data

        Raises:
            FileNotFoundError: If the CSV file doesn't exist
            ValueError: If required columns are missing
        """
        self.symbol = symbol
        self._filepath = Path(filepath)

        if not self._filepath.exists():
            raise FileNotFoundError(f"Data file not found: {self._filepath}")

        self._data = self._load_data()
        self._current_index = 0
        logger.debug(
            "Loaded %d bars for %s from %s",
            len(self._data),
            self.symbol,
            self._filepath.name,
        )

    def __repr__(self) -> str:
        return (
            f"CSVDataHandler(symbol={self.symbol!r}, "
            f"bars={len(self._data)}, "
            f"current_index={self._current_index})"
        )

    def _load_data(self) -> pd.DataFrame:
        """Load and validate CSV data."""
        df = pd.read_csv(self._filepath)

        # Handle different column naming conventions
        df.columns = df.columns.str.strip().str.lower()

        # Rename common variations
        column_mapping = {
            "date": "timestamp",
            "datetime": "timestamp",
            "adj close": "adj_close",
        }
        df = df.rename(columns=column_mapping)

        required_columns = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Parse timestamp and sort
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df
