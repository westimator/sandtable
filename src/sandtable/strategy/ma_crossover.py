"""
src/sandtable/strategy/ma_crossover.py

Moving average crossover strategy.

A classic trend-following strategy that generates signals when
fast and slow moving averages cross.
"""

from dataclasses import dataclass, field

from sandtable.core.events import Direction, MarketDataEvent, SignalEvent
from sandtable.strategy.abstract_strategy import AbstractStrategy
from sandtable.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MACrossoverStrategy(AbstractStrategy):
    """
    Moving average crossover strategy.

    Generates LONG signals when the fast MA crosses above the slow MA
    (bullish crossover), and SHORT signals when the fast MA crosses
    below the slow MA (bearish crossover).

    Returns None during warmup period when there isn't enough data
    to calculate both moving averages.

    Attributes:
        fast_period: Period for the fast moving average (default 10)
        slow_period: Period for the slow moving average (default 30)
    """

    fast_period: int = 10
    slow_period: int = 30

    _prev_fast_ma: dict[str, float] = field(init=False, default_factory=dict)
    _prev_slow_ma: dict[str, float] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        """
        Initialize strategy state.
        """
        super().__post_init__()
        self._prev_fast_ma = {}
        self._prev_slow_ma = {}

        if self.fast_period >= self.slow_period:
            raise ValueError(
                f"fast_period ({self.fast_period}) must be less than "
                f"slow_period ({self.slow_period})"
            )

        logger.debug(
            "MACrossoverStrategy initialized: fast=%d, slow=%d",
            self.fast_period,
            self.slow_period,
        )

    def generate_signal(self, bar: MarketDataEvent) -> SignalEvent | None:
        """Generate signal based on MA crossover.

        Args:
            bar: Current market data bar

        Returns:
            SignalEvent on crossover, None during warmup or no crossover
        """
        symbol = bar.symbol

        # Check warmup period (per-symbol)
        if self.symbol_bar_count(symbol) < self.slow_period:
            logger.debug(
                "Warmup: %s %d/%d bars",
                symbol,
                self.symbol_bar_count(symbol),
                self.slow_period,
            )
            return None

        # Calculate current MAs (per-symbol)
        closes = self.get_historical_closes(self.slow_period, symbol)
        fast_ma = sum(closes[-self.fast_period :]) / self.fast_period
        slow_ma = sum(closes) / self.slow_period

        logger.debug(
            "%s MAs: fast(%d)=$%.2f, slow(%d)=$%.2f",
            symbol,
            self.fast_period,
            fast_ma,
            self.slow_period,
            slow_ma,
        )

        signal = None

        # Check for crossover (need previous values for this symbol)
        prev_fast = self._prev_fast_ma.get(symbol)
        prev_slow = self._prev_slow_ma.get(symbol)

        if prev_fast is not None and prev_slow is not None:
            # Bullish crossover: fast crosses above slow
            if prev_fast <= prev_slow and fast_ma > slow_ma:
                signal = SignalEvent(
                    timestamp=bar.timestamp,
                    symbol=symbol,
                    direction=Direction.LONG,
                    strength=1.0,
                )
                logger.debug(
                    "%s BULLISH crossover: fast($%.2f) crossed above slow($%.2f)",
                    symbol,
                    fast_ma,
                    slow_ma,
                )

            # Bearish crossover: fast crosses below slow
            elif prev_fast >= prev_slow and fast_ma < slow_ma:
                signal = SignalEvent(
                    timestamp=bar.timestamp,
                    symbol=symbol,
                    direction=Direction.SHORT,
                    strength=1.0,
                )
                logger.debug(
                    "%s BEARISH crossover: fast($%.2f) crossed below slow($%.2f)",
                    symbol,
                    fast_ma,
                    slow_ma,
                )

        # Store current MAs for next comparison (per-symbol)
        self._prev_fast_ma[symbol] = fast_ma
        self._prev_slow_ma[symbol] = slow_ma

        return signal

    def reset(self) -> None:
        """
        Reset strategy state for a new backtest run.
        """
        super().reset()
        self._prev_fast_ma.clear()
        self._prev_slow_ma.clear()
        logger.debug("MACrossoverStrategy reset")
