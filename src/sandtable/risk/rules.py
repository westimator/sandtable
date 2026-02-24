"""
src/sandtable/risk/rules.py

Individual, composable risk rules. Each has a check(order, portfolio) method
returning OrderEvent | None.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from sandtable.utils.logger import get_logger

if TYPE_CHECKING:
    from sandtable.core.events import OrderEvent
    from sandtable.portfolio.portfolio import Portfolio

logger = get_logger(__name__)


class AbstractRule(ABC):
    """
    Base class for individual risk rules.

    Each rule inspects a proposed order against the current portfolio and
    returns the order (possibly resized) or None to block it.

    After check() is called, breach_value and threshold hold the metric
    value that triggered the breach and the rule's configured limit. If
    check() passes without breach, both remain 0.0.
    """

    @abstractmethod
    def check(self, order: OrderEvent, portfolio: Portfolio) -> OrderEvent | None:
        """
        Evaluate a proposed order against this rule.

        Args:
            order: The proposed order event
            portfolio: Current portfolio state

        Returns:
            The (possibly resized) order, or None to block it.
        """


@dataclass
class MaxPositionSizeRule(AbstractRule):
    """
    No single position may exceed a given percentage of total equity.

    If the proposed order would push the position over the limit, the
    quantity is reduced to fit. If the reduced quantity is 0, the order
    is blocked entirely.
    """

    max_position_pct: float = 0.25
    breach_value: float = field(init=False, default=0.0, repr=False)
    threshold: float = field(init=False, default=0.0, repr=False)

    def check(self, order: OrderEvent, portfolio: Portfolio) -> OrderEvent | None:
        self.breach_value = 0.0
        self.threshold = 0.0
        equity = portfolio.equity()
        if equity <= 0:
            return None

        max_value = self.max_position_pct * equity
        price = portfolio._current_prices.get(order.symbol, 0.0)
        if price <= 0:
            return order

        pos = portfolio._positions.get(order.symbol)
        current_value = abs(pos.market_value(price)) if pos else 0.0

        proposed_value = order.quantity * price
        proposed_pct = (current_value + proposed_value) / equity

        if current_value + proposed_value <= max_value:
            return order

        self.breach_value = proposed_pct
        self.threshold = self.max_position_pct

        remaining = max_value - current_value
        if remaining <= 0:
            logger.debug(
                "MaxPositionSizeRule: blocking order for %s - already at limit",
                order.symbol,
            )
            return None

        new_qty = int(remaining / price)
        if new_qty <= 0:
            return None

        logger.debug(
            "MaxPositionSizeRule: reducing %s order from %d to %d shares",
            order.symbol,
            order.quantity,
            new_qty,
        )
        return replace(order, quantity=new_qty)


@dataclass
class MaxPortfolioExposureRule(AbstractRule):
    """
    Total gross exposure (sum of abs position values) must not exceed
    a given percentage of equity.
    """

    max_gross_exposure_pct: float = 1.0
    breach_value: float = field(init=False, default=0.0, repr=False)
    threshold: float = field(init=False, default=0.0, repr=False)

    def check(self, order: OrderEvent, portfolio: Portfolio) -> OrderEvent | None:
        self.breach_value = 0.0
        self.threshold = 0.0
        equity = portfolio.equity()
        if equity <= 0:
            return None

        max_exposure = self.max_gross_exposure_pct * equity

        # Current gross exposure across all positions
        current_exposure = 0.0
        for symbol, pos in portfolio._positions.items():
            price = portfolio._current_prices.get(symbol, 0.0)
            current_exposure += abs(pos.market_value(price))

        price = portfolio._current_prices.get(order.symbol, 0.0)
        if price <= 0:
            return order

        proposed_exposure = order.quantity * price
        proposed_pct = (current_exposure + proposed_exposure) / equity

        if current_exposure + proposed_exposure <= max_exposure:
            return order

        self.breach_value = proposed_pct
        self.threshold = self.max_gross_exposure_pct

        remaining = max_exposure - current_exposure
        if remaining <= 0:
            logger.debug(
                "MaxPortfolioExposureRule: blocking order - gross exposure at limit",
            )
            return None

        new_qty = int(remaining / price)
        if new_qty <= 0:
            return None

        logger.debug(
            "MaxPortfolioExposureRule: reducing %s order from %d to %d shares",
            order.symbol, order.quantity, new_qty,
        )
        return replace(order, quantity=new_qty)


@dataclass
class MaxDrawdownRule(AbstractRule):
    """
    If peak-to-trough drawdown exceeds a threshold, halt all new orders.
    """

    max_drawdown_pct: float = 0.20
    breach_value: float = field(init=False, default=0.0, repr=False)
    threshold: float = field(init=False, default=0.0, repr=False)

    def check(self, order: OrderEvent, portfolio: Portfolio) -> OrderEvent | None:
        self.breach_value = 0.0
        self.threshold = 0.0
        curve = portfolio._equity_curve
        if not curve:
            return order

        peak = max(p.equity for p in curve)
        current = curve[-1].equity

        if peak <= 0:
            return order

        drawdown = (peak - current) / peak

        if drawdown >= self.max_drawdown_pct:
            self.breach_value = drawdown
            self.threshold = self.max_drawdown_pct
            logger.debug(
                "MaxDrawdownRule: blocking order - drawdown %.2f%% >= limit %.2f%%",
                drawdown * 100, self.max_drawdown_pct * 100,
            )
            return None

        return order


@dataclass
class MaxDailyLossRule(AbstractRule):
    """
    If today's loss exceeds a percentage of start-of-day equity, block
    further trading for the day.
    """

    max_daily_loss_pct: float = 0.02
    breach_value: float = field(init=False, default=0.0, repr=False)
    threshold: float = field(init=False, default=0.0, repr=False)

    def check(self, order: OrderEvent, portfolio: Portfolio) -> OrderEvent | None:
        self.breach_value = 0.0
        self.threshold = 0.0
        curve = portfolio._equity_curve
        if not curve:
            return order

        current = curve[-1].equity
        current_date = curve[-1].timestamp.date()

        # Find start-of-day equity (first equity point on current date)
        start_of_day = current
        for point in curve:
            if point.timestamp.date() == current_date:
                start_of_day = point.equity
                break

        if start_of_day <= 0:
            return order

        daily_loss = (start_of_day - current) / start_of_day

        if daily_loss >= self.max_daily_loss_pct:
            self.breach_value = daily_loss
            self.threshold = self.max_daily_loss_pct
            logger.debug(
                "MaxDailyLossRule: blocking order - daily loss %.2f%% >= limit %.2f%%",
                daily_loss * 100, self.max_daily_loss_pct * 100,
            )
            return None

        return order


@dataclass
class MaxLeverageRule(AbstractRule):
    """
    Gross exposure divided by equity must not exceed max_leverage.

    Resizes order to fit if possible, rejects if resize would be zero.
    """

    max_leverage: float = 2.0
    breach_value: float = field(init=False, default=0.0, repr=False)
    threshold: float = field(init=False, default=0.0, repr=False)

    def check(self, order: OrderEvent, portfolio: Portfolio) -> OrderEvent | None:
        self.breach_value = 0.0
        self.threshold = 0.0
        equity = portfolio.equity()
        if equity <= 0:
            return None

        max_exposure = self.max_leverage * equity
        current_exposure = portfolio.gross_exposure()

        price = portfolio._current_prices.get(order.symbol, 0.0)
        if price <= 0:
            return order

        proposed_exposure = order.quantity * price
        proposed_leverage = (current_exposure + proposed_exposure) / equity

        if current_exposure + proposed_exposure <= max_exposure:
            return order

        self.breach_value = proposed_leverage
        self.threshold = self.max_leverage

        remaining = max_exposure - current_exposure
        if remaining <= 0:
            logger.debug(
                "MaxLeverageRule: blocking order - leverage at limit",
            )
            return None

        new_qty = int(remaining / price)
        if new_qty <= 0:
            return None

        logger.debug(
            "MaxLeverageRule: reducing %s order from %d to %d shares",
            order.symbol, order.quantity, new_qty,
        )
        return replace(order, quantity=new_qty)


@dataclass
class MaxOrderSizeRule(AbstractRule):
    """
    Hard reject if order quantity exceeds max_order_qty.

    No resizing - this is a circuit breaker for fat-finger errors.
    """

    max_order_qty: int = 10_000
    breach_value: float = field(init=False, default=0.0, repr=False)
    threshold: float = field(init=False, default=0.0, repr=False)

    def check(self, order: OrderEvent, portfolio: Portfolio) -> OrderEvent | None:
        self.breach_value = 0.0
        self.threshold = 0.0
        if order.quantity > self.max_order_qty:
            self.breach_value = float(order.quantity)
            self.threshold = float(self.max_order_qty)
            logger.debug(
                "MaxOrderSizeRule: blocking order - %d shares > limit %d",
                order.quantity, self.max_order_qty,
            )
            return None
        return order


@dataclass
class MaxConcentrationRule(AbstractRule):
    """
    Position value for order.symbol as a fraction of gross exposure
    must not exceed max_concentration_pct.

    Resizes or rejects if the order would push concentration above the limit.
    """

    max_concentration_pct: float = 0.30
    breach_value: float = field(init=False, default=0.0, repr=False)
    threshold: float = field(init=False, default=0.0, repr=False)

    def check(self, order: OrderEvent, portfolio: Portfolio) -> OrderEvent | None:
        self.breach_value = 0.0
        self.threshold = 0.0
        price = portfolio._current_prices.get(order.symbol, 0.0)
        if price <= 0:
            return order

        # current gross exposure across all positions
        current_gross = portfolio.gross_exposure()

        # if no existing exposure, allow the first order (can't diversify from nothing)
        if current_gross <= 0:
            return order

        # current position value for this symbol
        pos = portfolio._positions.get(order.symbol)
        symbol_value = abs(pos.market_value(price)) if pos else 0.0

        proposed_value = order.quantity * price
        new_symbol_value = symbol_value + proposed_value
        new_gross = current_gross + proposed_value

        if new_gross <= 0:
            return order

        concentration = new_symbol_value / new_gross
        if concentration <= self.max_concentration_pct:
            return order

        self.breach_value = concentration
        self.threshold = self.max_concentration_pct

        # calculate max allowable symbol value
        # we need: (symbol_value + x * price) / (current_gross + x * price) <= max_pct
        # solving: symbol_value + x*price <= max_pct * (current_gross + x*price)
        # x*price * (1 - max_pct) <= max_pct * current_gross - symbol_value
        # x*price <= (max_pct * current_gross - symbol_value) / (1 - max_pct)
        denom = 1.0 - self.max_concentration_pct
        if denom <= 0:
            return order

        max_additional_value = (self.max_concentration_pct * current_gross - symbol_value) / denom
        if max_additional_value <= 0:
            logger.debug(
                "MaxConcentrationRule: blocking order for %s - concentration at limit",
                order.symbol,
            )
            return None

        new_qty = int(max_additional_value / price)
        if new_qty <= 0:
            return None

        logger.debug(
            "MaxConcentrationRule: reducing %s order from %d to %d shares",
            order.symbol, order.quantity, new_qty,
        )
        return replace(order, quantity=new_qty)
