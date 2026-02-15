"""
src/sandtable/execution/slippage.py

Slippage models for realistic execution simulation.

Slippage represents the difference between expected and actual fill prices,
typically working against the trader (buying higher, selling lower).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from sandtable.core.events import MarketDataEvent, OrderEvent
from sandtable.utils.logger import get_logger

logger = get_logger(__name__)


class SlippageModel(ABC):
    """
    Abstract base class for slippage models.
    """

    @abstractmethod
    def calculate_slippage(
        self,
        order: OrderEvent,
        bar: MarketDataEvent,
        base_price: float,
    ) -> float:
        """
        Calculate slippage amount for an order.

        Args:
            order: The order being executed
            bar: Current market data bar
            base_price: The base execution price before slippage

        Returns:
            Slippage amount in price units (always positive, applied adversely)
        """
        pass


@dataclass
class ZeroSlippage(SlippageModel):
    """
    No slippage - fills execute at the base price (close).

    Use as a baseline for comparing other slippage models.
    """

    def calculate_slippage(
        self,
        order: OrderEvent,
        bar: MarketDataEvent,
        base_price: float,
    ) -> float:
        """
        Return zero slippage.
        """
        logger.debug("ZeroSlippage: no slippage applied")
        return 0.0


@dataclass
class FixedSlippage(SlippageModel):
    """
    Fixed slippage in basis points (bps) against the trader.

    Attributes:
        bps: Slippage in basis points (1 bp = 0.01% = 0.0001)
    """

    bps: float = 5.0

    def calculate_slippage(
        self,
        order: OrderEvent,
        bar: MarketDataEvent,
        base_price: float,
    ) -> float:
        """
        Calculate fixed basis points slippage.

        Returns:
            Slippage amount = base_price * (bps / 10000)
        """
        slippage = base_price * (self.bps / 10000.0)
        logger.debug(
            "FixedSlippage: %.2f bps on $%.2f = $%.4f",
            self.bps,
            base_price,
            slippage,
        )
        return slippage


@dataclass
class SpreadSlippage(SlippageModel):
    """
    Slippage based on the bar's high-low range as a spread proxy.

    Uses a fraction of the bar's range to estimate half the bid-ask spread.
    This provides more realistic slippage that varies with market conditions.

    Attributes:
        spread_fraction: Fraction of (high - low) to use as slippage (default 0.5)
    """

    spread_fraction: float = 0.5

    def calculate_slippage(
        self,
        order: OrderEvent,
        bar: MarketDataEvent,
        base_price: float,
    ) -> float:
        """
        Calculate spread-based slippage.

        Uses half the spread (since we cross half the spread on each trade).

        Returns:
            Slippage amount = (high - low) * spread_fraction * 0.5
        """
        bar_range = bar.high - bar.low
        # Half spread because we only cross half on execution
        slippage = bar_range * self.spread_fraction * 0.5
        logger.debug(
            "SpreadSlippage: range=$%.2f, fraction=%.2f, slippage=$%.4f",
            bar_range,
            self.spread_fraction,
            slippage,
        )
        return slippage
