"""
src/sandtable/strategy/buy_and_hold_strategy.py

A simple buy and hold strategy. Useful as a benchmark.

Emits a single LONG signal on the first bar for each symbol,
then never signals again.
"""

from sandtable.core.events import Direction, MarketDataEvent, SignalEvent
from sandtable.strategy.abstract_strategy import AbstractStrategy
from sandtable.utils.logger import get_logger

logger = get_logger(__name__)


class BuyAndHoldStrategy(AbstractStrategy):
    """
    Buy and hold benchmark strategy.

    Generates a single LONG signal on the first bar received for each
    symbol, then stays silent. No parameters to tune.
    """

    def __init__(self, *, max_history: int = 500) -> None:
        super().__init__(max_history=max_history)
        self._entered: set[str] = set()

    def generate_signal(self, bar: MarketDataEvent) -> SignalEvent | None:
        """
        Generate a LONG signal on the first bar per symbol, None thereafter.

        Args:
            bar: Current market data bar

        Returns:
            SignalEvent(LONG) on first bar per symbol, None otherwise
        """
        if bar.symbol in self._entered:
            return None

        logger.debug("%s first bar, emitting BUY signal", bar.symbol)
        self._entered.add(bar.symbol)
        return SignalEvent(
            timestamp=bar.timestamp,
            symbol=bar.symbol,
            direction=Direction.LONG,
            strength=1.0,
        )

    def reset(self) -> None:
        """
        Reset strategy state for a new backtest run.
        """
        super().reset()
        self._entered.clear()
        logger.debug("BuyAndHoldStrategy reset")
