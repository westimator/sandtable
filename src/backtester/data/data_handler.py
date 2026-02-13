"""
src/backtester/data/data_handler.py

Data handler for loading and iterating over OHLCV data.

Implements strict index-based iteration to prevent look-ahead bias.
"""

import logging
from pathlib import Path

import pandas as pd

from backtester.core.events import MarketDataEvent

logger = logging.getLogger(__name__)


class CSVDataHandler:
    """
    Handles loading and iteration of OHLCV data from CSV files.

    Enforces temporal causality by only allowing access to historical data
    up to (but not including) the current bar index.

    Attributes:
        symbol: The ticker symbol for this data
        data: DataFrame containing OHLCV data sorted by timestamp

    Example:
        >>> handler = CSVDataHandler("data/spy.csv", "SPY")
        >>> while (bar := handler.get_next_bar()) is not None:
        ...     historical = handler.get_historical_bars(20)
        ...     process(bar, historical)
    """

    ## Magic methods 

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

        self._load_data()
        self._current_index = 0
        logger.info(
            "Loaded %d bars for %s from %s",
            len(self._data),
            self.symbol,
            self._filepath.name,
        )

    def __len__(self) -> int:
        """
        Return the total number of bars.
        """
        return len(self._data)

    def __repr__(self) -> str:
        return (
            f"CSVDataHandler(symbol={self.symbol!r}, "
            f"bars={len(self._data)}, "
            f"current_index={self._current_index})"
        )

    ## Private methods 

    def _load_data(self) -> None:
        """
        Load and validate CSV data.
        """
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

        self._data = df

    ## Properties 

    @property
    def data(self) -> pd.DataFrame:
        """
        Return the underlying DataFrame (read-only access).
        """
        return self._data.copy()

    @property
    def current_index(self) -> int:
        """
        Return the current bar index.
        """
        return self._current_index

    ## Public methods 

    def get_next_bar(self) -> MarketDataEvent | None:
        """
        Advance to the next bar and return it as a MarketDataEvent.

        Returns:
            MarketDataEvent for the next bar, or None if no more data.
        """
        if self._current_index >= len(self._data):
            logger.debug("No more bars available")
            return None

        row = self._data.iloc[self._current_index]
        self._current_index += 1

        event = MarketDataEvent(
            timestamp=row["timestamp"].to_pydatetime(),
            symbol=self.symbol,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )

        logger.debug(
            "Bar %d: %s %s O=%.2f H=%.2f L=%.2f C=%.2f V=%.0f",
            self._current_index,
            self.symbol,
            event.timestamp.date(),
            event.open,
            event.high,
            event.low,
            event.close,
            event.volume,
        )

        return event

    def get_historical_bars(self, n: int) -> list[MarketDataEvent]:
        """
        Get the n most recent bars BEFORE the current index.

        This method enforces look-ahead prevention by only returning
        data that would have been available at the current point in time.

        Args:
            n: Number of historical bars to retrieve

        Returns:
            List of MarketDataEvents, oldest first. May contain fewer
            than n bars if not enough history is available.
        """
        # Current index points to the NEXT bar to be read
        # So historical data is everything before current_index
        end_idx = self._current_index
        start_idx = max(0, end_idx - n)

        if end_idx == 0:
            logger.debug("No historical bars available (at start)")
            return []

        bars = []
        for idx in range(start_idx, end_idx):
            row = self._data.iloc[idx]
            bars.append(
                MarketDataEvent(
                    timestamp=row["timestamp"].to_pydatetime(),
                    symbol=self.symbol,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )

        logger.debug(
            "Retrieved %d historical bars (requested %d)",
            len(bars),
            n,
        )

        return bars

    def get_latest_bar(self) -> MarketDataEvent | None:
        """
        Get the most recent bar that has been read (the last bar returned by get_next_bar).

        Returns:
            The most recent MarketDataEvent, or None if no bars have been read.
        """
        if self._current_index == 0:
            return None

        row = self._data.iloc[self._current_index - 1]
        return MarketDataEvent(
            timestamp=row["timestamp"].to_pydatetime(),
            symbol=self.symbol,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )

    def reset(self) -> None:
        """
        Reset the data handler to the beginning for rerunning backtests.
        """
        self._current_index = 0
        logger.info("Data handler reset to beginning")
