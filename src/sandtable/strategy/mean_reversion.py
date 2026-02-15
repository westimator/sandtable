"""
src/sandtable/strategy/mean_reversion.py

Mean reversion strategy.

Generates signals when price deviates significantly from its
moving average, betting on a return to the mean.
"""

from dataclasses import dataclass

from sandtable.core.events import Direction, MarketDataEvent, SignalEvent
from sandtable.strategy.abstract_strategy import AbstractStrategy
from sandtable.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MeanReversionStrategy(AbstractStrategy):
    """
    Mean reversion strategy based on z-score of price relative to moving average.

    Generates LONG signals when the z-score drops below -threshold
    (price is unusually low) and SHORT signals when it rises above
    +threshold (price is unusually high).

    Signal strength scales with the magnitude of the deviation,
    capped at 1.0.

    Attributes:
        lookback: Period for the moving average and std dev (default 20)
        threshold: Z-score threshold to trigger signals (default 2.0)
    """

    lookback: int = 20
    threshold: float = 2.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.lookback < 2:
            raise ValueError(f"lookback ({self.lookback}) must be >= 2")
        if self.threshold <= 0:
            raise ValueError(f"threshold ({self.threshold}) must be > 0")
        logger.debug(
            "MeanReversionStrategy initialized: lookback=%d, threshold=%.1f",
            self.lookback,
            self.threshold,
        )

    def generate_signal(self, bar: MarketDataEvent) -> SignalEvent | None:
        """Generate signal based on z-score of price vs moving average.

        Args:
            bar: Current market data bar

        Returns:
            SignalEvent when z-score exceeds threshold, None otherwise
        """
        closes = self.get_historical_closes(self.lookback, bar.symbol)
        if len(closes) < self.lookback:
            return None

        mean = sum(closes) / len(closes)
        std = (sum((c - mean) ** 2 for c in closes) / len(closes)) ** 0.5

        if std == 0:
            return None

        z_score = (bar.close - mean) / std

        if z_score < -self.threshold:
            logger.debug(
                "%s LONG signal: z_score=%.2f < -%.1f",
                bar.symbol, z_score, self.threshold,
            )
            return SignalEvent(
                timestamp=bar.timestamp,
                symbol=bar.symbol,
                direction=Direction.LONG,
                strength=min(abs(z_score) / self.threshold, 1.0),
            )
        elif z_score > self.threshold:
            logger.debug(
                "%s SHORT signal: z_score=%.2f > %.1f",
                bar.symbol, z_score, self.threshold,
            )
            return SignalEvent(
                timestamp=bar.timestamp,
                symbol=bar.symbol,
                direction=Direction.SHORT,
                strength=min(abs(z_score) / self.threshold, 1.0),
            )
        return None
