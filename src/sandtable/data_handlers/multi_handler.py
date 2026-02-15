"""
src/sandtable/data_handlers/multi_handler.py

Multi-symbol data handler that merges bars from multiple single-symbol handlers.

Uses a min-heap to emit bars in chronological order across all symbols,
working transparently with the existing event loop.
"""

import heapq
from typing import Any

import pandas as pd

from sandtable.core.events import MarketDataEvent
from sandtable.data_handlers.abstract_data_handler import AbstractDataHandler
from sandtable.data_handlers.single_symbol_handler import SingleSymbolDataHandler
from sandtable.utils.logger import get_logger

logger = get_logger(__name__)


class MultiDataHandler(AbstractDataHandler):
    """
    Wraps multiple SingleSymbolDataHandlers and merges their bars
    in timestamp order using a min-heap.

    Attributes:
        symbol: Comma-separated list of symbols (for display)
    """

    def __init__(self, handlers: dict[str, SingleSymbolDataHandler]) -> None:
        if not handlers:
            raise ValueError("At least one handler is required")
        for sym, handler in handlers.items():
            if isinstance(handler, MultiDataHandler):
                raise TypeError(
                    f"Cannot nest MultiDataHandler (got one for key {sym!r})"
                )

        self._handlers = handlers
        self.symbol = ",".join(sorted(handlers.keys()))
        self._heap: list[tuple[Any, ...]] = []
        self._counter = 0
        self._current_index = 0

        # seed the heap with the first bar from each handler
        for sym, _ in self._handlers.items():
            self._push_next(sym)

        logger.debug(
            "MultiDataHandler initialized with %d symbols: %s",
            len(handlers),
            self.symbol,
        )

    def __len__(self) -> int:
        return sum(len(h) for h in self._handlers.values())

    def __repr__(self) -> str:
        return (
            f"MultiDataHandler(symbols=[{self.symbol}], "
            f"total_bars={len(self)}, "
            f"current_index={self._current_index})"
        )

    def _push_next(self, symbol: str) -> None:
        """Pop the next bar from a handler and push it onto the heap."""
        handler = self._handlers[symbol]
        bar = handler.get_next_bar()
        if bar is not None:
            heapq.heappush(
                self._heap,
                (bar.timestamp, self._counter, bar),
            )
            self._counter += 1

    @property
    def data(self) -> pd.DataFrame:
        """Return a DataFrame with all bars from all handlers."""
        frames = []
        for sym, handler in self._handlers.items():
            df = handler.data.copy()
            df["symbol"] = sym
            frames.append(df)
        return pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    def get_next_bar(self) -> MarketDataEvent | None:
        if not self._heap:
            return None

        _ts, _cnt, bar = heapq.heappop(self._heap)
        self._current_index += 1

        # Refill from the same handler
        self._push_next(bar.symbol)

        return bar

    def get_historical_bars(self, n: int) -> list[MarketDataEvent]:
        # Delegate to individual handlers and merge
        all_bars: list[MarketDataEvent] = []
        for handler in self._handlers.values():
            all_bars.extend(handler.get_historical_bars(n))
        all_bars.sort(key=lambda b: b.timestamp)
        return all_bars[-n:] if len(all_bars) > n else all_bars

    def get_latest_bar(self) -> MarketDataEvent | None:
        latest = None
        for handler in self._handlers.values():
            bar = handler.get_latest_bar()
            if bar is not None and (latest is None or bar.timestamp > latest.timestamp):
                latest = bar
        return latest

    def reset(self) -> None:
        """Reset the data handler to the beginning for rerunning backtests."""
        for handler in self._handlers.values():
            handler.reset()
        self._heap = []
        self._counter = 0
        self._current_index = 0
        for sym in self._handlers:
            self._push_next(sym)
        logger.debug("MultiDataHandler reset")

    def get_price_data(self) -> dict[str, pd.DataFrame]:
        """Return price data keyed by symbol."""
        return {sym: h.data for sym, h in self._handlers.items()}
