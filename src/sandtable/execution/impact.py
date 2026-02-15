"""
src/sandtable/execution/impact.py

Market impact models for realistic execution simulation.

Market impact represents the price movement caused by the order itself,
particularly relevant for larger orders relative to average daily volume.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from sandtable.core.events import MarketDataEvent, OrderEvent
from sandtable.utils.logger import get_logger

logger = get_logger(__name__)


class MarketImpactModel(ABC):
    """
    Abstract base class for market impact models.
    """

    @abstractmethod
    def calculate_impact(
        self,
        order: OrderEvent,
        bar: MarketDataEvent,
        base_price: float,
    ) -> float:
        """
        Calculate market impact for an order.

        Args:
            order: The order being executed
            bar: Current market data bar (contains volume for ADV proxy)
            base_price: The base execution price

        Returns:
            Impact amount in price units (always positive, applied adversely)
        """
        pass


@dataclass
class NoMarketImpact(MarketImpactModel):
    """
    No market impact (order size doesn't affect price).

    Use as a baseline for comparing other impact models.
    """

    def calculate_impact(
        self,
        order: OrderEvent,
        bar: MarketDataEvent,
        base_price: float,
    ) -> float:
        """
        Return zero market impact.
        """
        logger.debug("NoMarketImpact: no impact applied")
        return 0.0


@dataclass
class SquareRootImpactModel(MarketImpactModel):
    """
    Square-root market impact model.

    Based on the widely-used square-root law of market impact:
        impact = sigma * sqrt(order_size / ADV) * eta

    Where:
        - sigma: Daily volatility (estimated from bar range or provided)
        - order_size: Number of shares in the order
        - ADV: Average daily volume (approximated by current bar volume)
        - eta: Impact coefficient (calibration parameter)

    Attributes:
        eta: Impact coefficient (default 0.1)
        sigma: Daily volatility as decimal (default None = estimate from bar)
    """

    eta: float = 0.1
    sigma: float | None = None

    def calculate_impact(
        self,
        order: OrderEvent,
        bar: MarketDataEvent,
        base_price: float,
    ) -> float:
        """
        Calculate square-root market impact.

        If sigma is not provided, estimates volatility from the bar's
        high-low range using the Parkinson estimator approximation.

        Returns:
            Impact amount = base_price * sigma * sqrt(order_size / ADV) * eta
        """
        # Use provided sigma or estimate from bar range
        if self.sigma is not None:
            volatility = self.sigma
        else:
            # Parkinson volatility estimator (simplified)
            # sigma ≈ (high - low) / (2 * sqrt(ln(2)) * price) ≈ 0.6 * range / price
            if bar.close > 0:
                volatility = 0.6 * (bar.high - bar.low) / bar.close
            else:
                volatility = 0.01  # fallback

        # Use bar volume as ADV proxy (in production, use rolling average)
        adv = bar.volume if bar.volume > 0 else 1.0

        # Participation rate
        participation = order.quantity / adv

        # Square-root impact
        if participation > 0:
            impact = base_price * volatility * math.sqrt(participation) * self.eta
        else:
            impact = 0.0

        logger.debug(
            "SquareRootImpact: sigma=%.4f, qty=%d, ADV=%.0f, participation=%.6f, impact=$%.4f",
            volatility,
            order.quantity,
            adv,
            participation,
            impact,
        )

        return impact
