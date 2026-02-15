"""
src/sandtable/data_handlers/abstract_data_handler.py

AbstractDataHandler ABC — the interface all data handlers must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from sandtable.core.events import MarketDataEvent


class AbstractDataHandler(ABC):
    """ABC defining the interface for all data handlers.

    Both single-symbol and multi-symbol handlers implement this interface.
    """

    symbol: str
    _current_index: int

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of bars."""

    @property
    @abstractmethod
    def data(self) -> pd.DataFrame:
        """Return the underlying OHLCV data."""

    @property
    def current_index(self) -> int:
        """Return the current bar index."""
        return self._current_index

    @abstractmethod
    def get_next_bar(self) -> MarketDataEvent | None:
        """Advance to the next bar and return it as a MarketDataEvent."""

    @abstractmethod
    def get_historical_bars(self, n: int) -> list[MarketDataEvent]:
        """Get the n most recent bars BEFORE the current index."""

    @abstractmethod
    def get_latest_bar(self) -> MarketDataEvent | None:
        """Get the most recent bar that has been read."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the data handler to the beginning for rerunning backtests."""

    def get_price_data(self) -> dict[str, pd.DataFrame]:
        """Return price data keyed by symbol."""
        return {self.symbol: self.data}
