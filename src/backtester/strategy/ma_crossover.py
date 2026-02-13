"""
src/backtester/strategy/ma_crossover.py

Moving average crossover strategy.

A classic trend-following strategy that generates signals when
fast and slow moving averages cross.
"""

import logging
from dataclasses import dataclass, field

from backtester.core.events import Direction, MarketDataEvent, SignalEvent
from backtester.strategy.base import Strategy

logger = logging.getLogger(__name__)


@dataclass
class MACrossoverStrategy(Strategy):
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

    _prev_fast_ma: float | None = field(init=False, default=None)
    _prev_slow_ma: float | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        """
        Initialize strategy state.
        """
        super().__post_init__()
        self._prev_fast_ma = None
        self._prev_slow_ma = None

        if self.fast_period >= self.slow_period:
            raise ValueError(
                f"fast_period ({self.fast_period}) must be less than "
                f"slow_period ({self.slow_period})"
            )

        logger.info(
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
        # Check warmup period
        if self.bar_count < self.slow_period:
            logger.debug(
                "Warmup: %d/%d bars",
                self.bar_count,
                self.slow_period,
            )
            return None

        # Calculate current MAs
        closes = self.get_historical_closes(self.slow_period)
        fast_ma = sum(closes[-self.fast_period :]) / self.fast_period
        slow_ma = sum(closes) / self.slow_period

        logger.debug(
            "MAs: fast(%.2f)=$%.2f, slow(%d)=$%.2f",
            self.fast_period,
            fast_ma,
            self.slow_period,
            slow_ma,
        )

        signal = None

        # Check for crossover (need previous values)
        if self._prev_fast_ma is not None and self._prev_slow_ma is not None:
            # Bullish crossover: fast crosses above slow
            if self._prev_fast_ma <= self._prev_slow_ma and fast_ma > slow_ma:
                signal = SignalEvent(
                    timestamp=bar.timestamp,
                    symbol=bar.symbol,
                    direction=Direction.LONG,
                    strength=1.0,
                )
                logger.info(
                    "BULLISH crossover: fast($%.2f) crossed above slow($%.2f)",
                    fast_ma,
                    slow_ma,
                )

            # Bearish crossover: fast crosses below slow
            elif self._prev_fast_ma >= self._prev_slow_ma and fast_ma < slow_ma:
                signal = SignalEvent(
                    timestamp=bar.timestamp,
                    symbol=bar.symbol,
                    direction=Direction.SHORT,
                    strength=1.0,
                )
                logger.info(
                    "BEARISH crossover: fast($%.2f) crossed below slow($%.2f)",
                    fast_ma,
                    slow_ma,
                )

        # Store current MAs for next comparison
        self._prev_fast_ma = fast_ma
        self._prev_slow_ma = slow_ma

        return signal

    def reset(self) -> None:
        """
        Reset strategy state for a new backtest run.
        """
        super().reset()
        self._prev_fast_ma = None
        self._prev_slow_ma = None
        logger.debug("MACrossoverStrategy reset")
