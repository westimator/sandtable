"""
src/sandtable/data_handlers/single_symbol_handler.py

SingleSymbolDataHandler — shared OHLCV iteration logic for handlers
that serve a single ticker symbol.

Subclasses must set ``symbol``, ``_data``, and ``_current_index`` in their ``__init__``.
"""

from __future__ import annotations

from abc import ABC

import pandas as pd

from sandtable.core.events import MarketDataEvent
from sandtable.data_handlers.abstract_data_handler import AbstractDataHandler
from sandtable.utils.logger import get_logger

logger = get_logger(__name__)


class SingleSymbolDataHandler(AbstractDataHandler, ABC):
    """
    Shared iteration logic for single-symbol data handlers.

    Subclasses must set these attributes in __init__:
        symbol: str, ticker symbol
        _data: pd.DataFrame, OHLCV data with 'timestamp', 'open', 'high',
                               'low', 'close', 'volume' columns
        _current_index: int, should start at 0
    """

    _data: pd.DataFrame

    def __len__(self) -> int:
        """Return the total number of bars."""
        return len(self._data)

    @property
    def data(self) -> pd.DataFrame:
        """Return the underlying DataFrame (read-only copy)."""
        return self._data.copy()

    def _row_to_event(self, row: pd.Series) -> MarketDataEvent:
        """Convert a DataFrame row to a MarketDataEvent."""
        return MarketDataEvent(
            timestamp=row["timestamp"].to_pydatetime(),
            symbol=self.symbol,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )

    def get_next_bar(self) -> MarketDataEvent | None:
        """Advance to the next bar and return it as a MarketDataEvent."""
        if self._current_index >= len(self._data):
            logger.debug("Data exhausted for %s at index %d", self.symbol, self._current_index)
            return None

        row = self._data.iloc[self._current_index]
        self._current_index += 1
        return self._row_to_event(row)

    def get_historical_bars(self, n: int) -> list[MarketDataEvent]:
        """Get the n most recent bars BEFORE the current index.

        Returns bars oldest-first. May contain fewer than n bars
        if not enough history is available.
        """
        end_idx = self._current_index
        start_idx = max(0, end_idx - n)

        if end_idx == 0:
            return []

        return [
            self._row_to_event(self._data.iloc[i])
            for i in range(start_idx, end_idx)
        ]

    def get_latest_bar(self) -> MarketDataEvent | None:
        """Get the most recent bar that has been read."""
        if self._current_index == 0:
            return None
        return self._row_to_event(self._data.iloc[self._current_index - 1])

    def reset(self) -> None:
        """Reset the data handler to the beginning for rerunning backtests."""
        self._current_index = 0
        logger.debug("Reset %s to index 0", self.symbol)
