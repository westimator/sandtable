"""
src/sandtable/risk/risk_manager.py

Composable risk manager that chains multiple rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sandtable.core.events import RiskAction, RiskBreachEvent
from sandtable.risk.abstract_risk_manager import AbstractRiskManager
from sandtable.risk.rules import AbstractRule
from sandtable.utils.logger import get_logger

if TYPE_CHECKING:
    from sandtable.core.events import OrderEvent, SignalEvent
    from sandtable.portfolio.portfolio import Portfolio

logger = get_logger(__name__)


@dataclass
class RiskManager(AbstractRiskManager):
    """
    Composable risk manager that chains multiple rules.

    Each rule's check() method is called in sequence. If any rule returns
    None, the order is blocked and subsequent rules are skipped.
    Rejections and resizes are logged as RiskBreachEvents in breach_log.
    """

    rules: list[AbstractRule] = field(default_factory=list)
    breach_log: list[RiskBreachEvent] = field(default_factory=list)

    def evaluate(
        self,
        signal: SignalEvent,
        proposed_order: OrderEvent,
        portfolio: Portfolio,
    ) -> OrderEvent | None:
        """
        Evaluate a proposed order against all risk rules in sequence.

        Args:
            signal: The originating signal event
            proposed_order: The order produced by portfolio position sizing
            portfolio: Current portfolio state

        Returns:
            The (possibly resized) order, or None if any rule blocks it.
        """
        order = proposed_order
        for rule in self.rules:
            result = rule.check(order, portfolio)
            rule_name = type(rule).__name__

            if result is None:
                # rejected
                self.breach_log.append(RiskBreachEvent(
                    timestamp=proposed_order.timestamp,
                    rule_name=rule_name,
                    symbol=proposed_order.symbol,
                    proposed_qty=order.quantity,
                    action=RiskAction.REJECTED,
                    breach_value=getattr(rule, "breach_value", 0.0),
                    threshold=getattr(rule, "threshold", 0.0),
                    final_qty=None,
                ))
                logger.debug(
                    "Order for %s blocked by %s",
                    proposed_order.symbol,
                    rule_name,
                )
                return None

            if result.quantity != order.quantity:
                # resized
                self.breach_log.append(RiskBreachEvent(
                    timestamp=proposed_order.timestamp,
                    rule_name=rule_name,
                    symbol=proposed_order.symbol,
                    proposed_qty=order.quantity,
                    action=RiskAction.RESIZED,
                    breach_value=getattr(rule, "breach_value", 0.0),
                    threshold=getattr(rule, "threshold", 0.0),
                    final_qty=result.quantity,
                ))
                logger.debug(
                    "Order for %s resized by %s: %d -> %d",
                    proposed_order.symbol,
                    rule_name,
                    order.quantity,
                    result.quantity,
                )

            order = result
        return order
