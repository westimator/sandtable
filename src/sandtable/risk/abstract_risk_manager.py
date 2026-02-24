"""
src/sandtable/risk/abstract_risk_manager.py

Abstract base class for risk managers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sandtable.core.events import OrderEvent, SignalEvent
    from sandtable.portfolio.portfolio import Portfolio


class AbstractRiskManager(ABC):
    """
    Base class for risk managers that filter proposed orders.

    A risk manager evaluates a proposed order against risk constraints
    and returns the order (possibly resized) or None to block it.
    """

    @abstractmethod
    def evaluate(
        self,
        signal: SignalEvent,
        proposed_order: OrderEvent,
        portfolio: Portfolio,
    ) -> OrderEvent | None:
        """
        Evaluate a proposed order against risk constraints.

        Args:
            signal: The originating signal event
            proposed_order: The order produced by portfolio position sizing
            portfolio: Current portfolio state

        Returns:
            The (possibly resized) order, or None to block it.
        """
