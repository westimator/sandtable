"""
src/sandtable/strategy/abstract_strategy.py

Abstract base class for trading strategies.

Strategies receive market data events and generate trading signals.
"""

from abc import ABC, abstractmethod
from collections import deque

from sandtable.core.events import MarketDataEvent, SignalEvent
from sandtable.utils.logger import get_logger

logger = get_logger(__name__)


class AbstractStrategy(ABC):
    """
    Abstract base class for trading strategies.

    Strategies maintain a history of market data bars and generate
    trading signals based on their specific logic.

    Subclasses must implement `generate_signal()` to define the
    trading logic.

    Attributes:
        max_history: Maximum number of bars to keep in history
    """

    ## Magic methods

    def __init__(self, *, max_history: int = 500) -> None:
        """
        Initialize the strategy.

        Args:
            max_history: Maximum number of bars to keep in history
        """
        self.max_history = max_history
        self._bar_history: deque[MarketDataEvent] = deque(maxlen=max_history)

    ## Properties

    @property
    def bar_count(self) -> int:
        """
        Number of bars in history.
        """
        return len(self._bar_history)

    @property
    def bars(self) -> list[MarketDataEvent]:
        """
        Get all bars in history (oldest first).
        """
        return list(self._bar_history)

    ## Abstract methods

    @abstractmethod
    def generate_signal(self, bar: MarketDataEvent) -> SignalEvent | None:
        """
        Generate a trading signal based on current market state.

        Called by on_bar() after the new bar is added to history.
        Subclasses implement their trading logic here.

        Args:
            bar: The current (most recent) market data event

        Returns:
            A SignalEvent if the strategy wants to trade, None otherwise.
        """
        pass

    ## Public API

    def on_bar(self, bar: MarketDataEvent) -> SignalEvent | None:
        """
        Process a new market data bar.

        Stores the bar in history and calls generate_signal().

        Args:
            bar: The new market data event

        Returns:
            A SignalEvent if the strategy wants to trade, None otherwise.
        """
        self._bar_history.append(bar)
        logger.debug(
            "Strategy received bar %d: %s %s close=$%.2f",
            self.bar_count, bar.symbol, bar.timestamp.date(), bar.close,
        )
        return self.generate_signal(bar)

    def symbol_bar_count(self, symbol: str) -> int:
        """Number of bars in history for a specific symbol."""
        return sum(
            1 for b in self._bar_history if b.symbol == symbol
        )

    def get_historical_closes(self, n: int, symbol: str | None = None) -> list[float]:
        """
        Get the n most recent closing prices.

        Args:
            n: Number of closes to retrieve
            symbol: If provided, only return closes for this symbol.
                Recommended for multi-asset strategies.

        Returns:
            List of closing prices, oldest first. May contain fewer
            than n prices if not enough history.
        """
        if symbol is not None:
            bars = [b for b in self._bar_history if b.symbol == symbol]
        else:
            bars = list(self._bar_history)
        if n >= len(bars):
            return [b.close for b in bars]
        return [b.close for b in bars[-n:]]

    def get_historical_bars(self, n: int, symbol: str | None = None) -> list[MarketDataEvent]:
        """Get the n most recent bars.

        Args:
            n: Number of bars to retrieve
            symbol: If provided, only return bars for this symbol.
                Recommended for multi-asset strategies.

        Returns:
            List of MarketDataEvents, oldest first. May contain fewer
            than n bars if not enough history.
        """
        if symbol is not None:
            bars = [b for b in self._bar_history if b.symbol == symbol]
        else:
            bars = list(self._bar_history)
        if n >= len(bars):
            return bars
        return bars[-n:]

    def reset(self) -> None:
        """
        Reset strategy state for a new backtest run.
        """
        self._bar_history.clear()
        logger.debug("Strategy reset")
