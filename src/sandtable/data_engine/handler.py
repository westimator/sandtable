"""
src/sandtable/data_engine/handler.py

DataHandler converts provider DataFrames into a chronologically ordered
stream of MarketDataEvents across all symbols in a universe.

This is the bridge between the DataProvider layer and the event-driven engine.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime

import pandas as pd

from sandtable.core.events import MarketDataEvent
from sandtable.data.universe import Universe
from sandtable.data_engine.data_providers import AbstractDataProvider
from sandtable.utils.logger import get_logger

logger = get_logger(__name__)


class DataHandler:
    """
    Fetches data for a universe of symbols and yields MarketDataEvents
    in strict chronological order.

    Supports both iterator-based access (iter_events) and pull-based access
    (get_next_bar) for compatibility with the backtest engine.

    Usage:
        provider = CSVProvider("data/fixtures")
        universe = Universe.from_symbols(["SPY", "AAPL"])
        handler = DataHandler(provider, universe)
        handler.load("2018-01-01", "2023-12-31")
        for event in handler.iter_events():
            process(event)
    """

    def __init__(
        self,
        provider: AbstractDataProvider,
        universe: Universe | list[str] | None = None,
    ) -> None:
        """
        Args:
            provider: AbstractDataProvider to fetch OHLCV data from.
            universe: Universe or list[str] defining which symbols to load.
                If a list[str] is passed, auto-converts to Universe via from_symbols().
        """
        if universe is None:
            universe = Universe.from_symbols([])
        elif isinstance(universe, list):
            universe = Universe.from_symbols(universe)

        self._provider = provider
        self._universe = universe
        self._data: dict[str, pd.DataFrame] = {}
        self._event_iter: Iterator[MarketDataEvent] | None = None
        self._current_index: int = 0
        self._total_bars: int = 0

    @property
    def universe(self) -> Universe:
        """Return the universe used by this handler."""
        return self._universe

    @property
    def symbol(self) -> str:
        """Return comma-separated symbol string for logging."""
        return ",".join(self._universe.symbols)

    @property
    def current_index(self) -> int:
        """Return how many bars have been consumed via get_next_bar."""
        return self._current_index

    def __len__(self) -> int:
        """Return total number of bars across all symbols."""
        return self._total_bars

    @property
    def data(self) -> dict[str, pd.DataFrame]:
        """Return the loaded data, keyed by symbol."""
        return dict(self._data)

    def get_price_data(self) -> dict[str, pd.DataFrame]:
        """Return price data keyed by symbol (alias for data property)."""
        return dict(self._data)

    def load(self, start_date: str, end_date: str) -> None:
        """
        Pre-fetch all data for the universe. Called once before backtest starts.

        Args:
            start_date: Start date as 'YYYY-MM-DD'
            end_date: End date as 'YYYY-MM-DD'
        """
        self._data.clear()
        self._event_iter = None
        self._current_index = 0
        self._total_bars = 0
        for symbol in self._universe.symbols:
            logger.debug(
                "Loading data for %s (%s to %s)",
                symbol, start_date, end_date,
            )
            df = self._provider.fetch(
                symbol, start_date, end_date,
            )
            self._data[symbol] = df
            self._total_bars += len(df)
            logger.debug(
                "Loaded %d bars for %s",
                len(df), symbol,
            )

    def get_next_bar(self) -> MarketDataEvent | None:
        """
        Pull-based access: return the next bar, or None if exhausted.
        Used by the backtest engine's event loop.
        """
        if self._event_iter is None:
            self._event_iter = self.iter_events()
        try:
            bar = next(self._event_iter)
            self._current_index += 1
            return bar
        except StopIteration:
            return None

    def reset(self) -> None:
        """Reset the iterator so the handler can be reused for another run."""
        self._event_iter = None
        self._current_index = 0

    def date_slice(self, start_date: str, end_date: str) -> DataHandler:
        """
        Return a new DataHandler with data filtered to [start_date, end_date].

        Does not re-fetch from the provider - slices already loaded DataFrames.
        Useful for walk-forward analysis where the same data is split
        into train/test windows.
        """
        new_handler = DataHandler(
            self._provider, self._universe,
        )
        for symbol, df in self._data.items():
            sliced = df.loc[start_date:end_date]
            new_handler._data[symbol] = sliced
            new_handler._total_bars += len(sliced)
        return new_handler

    def iter_events(self) -> Iterator[MarketDataEvent]:
        """
        Yield MarketDataEvents in chronological order across all symbols.

        For each date, events are emitted for all symbols that have data
        on that date. This handles symbols with different trading calendars
        (some may have gaps).

        This is a generator - memory usage is flat regardless of backtest duration.
        """
        if not self._data:
            return

        # collect all unique dates across all symbols, sorted
        all_dates: set[datetime] = set()
        for df in self._data.values():
            all_dates.update(df.index.to_pydatetime())
        sorted_dates = sorted(all_dates)

        # for each date, emit events for every symbol that has data
        for date in sorted_dates:
            for symbol in self._universe.symbols:
                df = self._data.get(symbol)
                if df is None:
                    continue
                if date not in df.index:
                    continue

                row = df.loc[date]
                # handle case where date appears multiple times (shouldn't, but be safe)
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]

                yield MarketDataEvent(
                    timestamp=date,
                    symbol=symbol,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
