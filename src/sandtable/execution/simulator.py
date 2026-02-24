"""
src/sandtable/execution/simulator.py

Execution simulator for processing orders into fills.

Combines slippage models, market impact models, and commission calculations
to produce realistic fill events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sandtable.core.events import Direction, FillEvent, MarketDataEvent, OrderEvent
from sandtable.execution.impact import MarketImpactModel, NoMarketImpact
from sandtable.execution.slippage import SlippageModel, ZeroSlippage
from sandtable.utils.logger import get_logger

if TYPE_CHECKING:
    from sandtable.data.instrument import Instrument

logger = get_logger(__name__)


@dataclass
class ExecutionConfig:
    """
    Configuration for execution costs.

    Attributes:
        commission_per_share: Fixed commission per share (default $0.005)
        commission_pct: Commission as percentage of trade value (default 0.0)
        commission_minimum: Minimum commission per order (default $1.0)
    """

    commission_per_share: float = 0.005
    commission_pct: float = 0.0
    commission_minimum: float = 1.0


@dataclass
class ExecutionSimulator:
    """
    Simulates order execution with realistic costs.

    Processes OrderEvents and produces FillEvents with:
    - Slippage (price deviation from close)
    - Market impact (price movement from order size)
    - Commission (trading fees)

    All costs work against the trader:
    - Buys execute at higher prices
    - Sells execute at lower prices

    Fill prices are clamped to the bar's [low, high] range for realism.

    Attributes:
        config: Execution cost configuration
        slippage_model: Model for calculating slippage
        impact_model: Model for calculating market impact
    """

    config: ExecutionConfig = field(default_factory=ExecutionConfig)
    slippage_model: SlippageModel = field(default_factory=ZeroSlippage)
    impact_model: MarketImpactModel = field(default_factory=NoMarketImpact)

    def process_order(
        self,
        order: OrderEvent,
        bar: MarketDataEvent,
        instrument: Instrument | None = None,
    ) -> FillEvent:
        """
        Process an order and return a fill event.

        Args:
            order: The order to execute
            bar: Current market data bar for price/volume info
            instrument: Optional instrument for tick/lot size rounding

        Returns:
            FillEvent with execution details
        """
        # round order quantity to lot size if instrument provided
        quantity = order.quantity
        if instrument is not None and instrument.lot_size > 1:
            quantity = max(
                instrument.lot_size,
                (quantity // instrument.lot_size) * instrument.lot_size
            )

        # base price is the close (for market orders)
        base_price = bar.close

        # calculate slippage and impact (both positive values)
        slippage_amount = self.slippage_model.calculate_slippage(order, bar, base_price)
        impact_amount = self.impact_model.calculate_impact(order, bar, base_price)

        # apply costs adversely based on direction
        if order.direction == Direction.LONG:
            # buying: pay more (add slippage and impact)
            fill_price = base_price + slippage_amount + impact_amount
        else:
            # selling: receive less (subtract slippage and impact)
            fill_price = base_price - slippage_amount - impact_amount

        # round fill price to tick size if instrument provided
        if instrument is not None and instrument.tick_size > 0:
            fill_price = round(fill_price / instrument.tick_size) * instrument.tick_size

        # clamp fill price to bar's [low, high] range
        original_fill_price = fill_price
        fill_price = max(bar.low, min(bar.high, fill_price))

        if fill_price != original_fill_price:
            logger.debug(
                "Fill price clamped from $%.4f to $%.4f (bar range: $%.2f-$%.2f)",
                original_fill_price,
                fill_price,
                bar.low,
                bar.high,
            )

        # calculate commission
        commission = self._calculate_commission(quantity, fill_price)

        # create fill event
        fill = FillEvent(
            timestamp=order.timestamp,
            symbol=order.symbol,
            direction=order.direction,
            quantity=quantity,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage_amount,
            market_impact=impact_amount,
        )

        logger.debug(
            "Fill: %s %d %s @ $%.2f (slip=$%.4f, impact=$%.4f, comm=$%.2f)",
            order.direction.name,
            quantity,
            order.symbol,
            fill_price,
            slippage_amount,
            impact_amount,
            commission,
        )

        return fill

    def _calculate_commission(self, quantity: int, fill_price: float) -> float:
        """
        Calculate commission for a trade.

        Combines per-share and percentage-based commissions,
        enforcing a minimum commission.

        Args:
            quantity: Number of shares
            fill_price: Execution price per share

        Returns:
            Total commission amount
        """
        trade_value = quantity * fill_price

        # per-share commission
        per_share_comm = quantity * self.config.commission_per_share

        # percentage commission
        pct_comm = trade_value * self.config.commission_pct

        # total with minimum
        total_commission = max(
            per_share_comm + pct_comm,
            self.config.commission_minimum,
        )

        logger.debug(
            "Commission: per_share=$%.2f + pct=$%.2f = $%.2f (min=$%.2f)",
            per_share_comm,
            pct_comm,
            total_commission,
            self.config.commission_minimum,
        )

        return total_commission
