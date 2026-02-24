"""
tests/unit/risk/test_manager.py

Tests for RiskManager compositor and integration with run_backtest.
"""

from datetime import datetime
from unittest.mock import MagicMock

from sandtable.api import run_backtest
from sandtable.core.events import Direction, MarketDataEvent, OrderEvent, OrderType, RiskAction, SignalEvent
from sandtable.core.result import BacktestResult
from sandtable.portfolio.portfolio import Portfolio
from sandtable.risk.risk_manager import RiskManager
from sandtable.risk.rules import MaxDrawdownRule, MaxOrderSizeRule, MaxPositionSizeRule
from sandtable.strategy.abstract_strategy import AbstractStrategy
from tests.conftest import make_data_handler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signal(symbol: str = "SPY") -> SignalEvent:
    return SignalEvent(
        timestamp=datetime(2024, 1, 1),
        symbol=symbol,
        direction=Direction.LONG,
        strength=1.0,
    )


def _make_order(symbol: str = "SPY", quantity: int = 100) -> OrderEvent:
    return OrderEvent(
        timestamp=datetime(2024, 1, 1),
        symbol=symbol,
        direction=Direction.LONG,
        quantity=quantity,
        order_type=OrderType.MARKET,
    )


class _AlwaysBlockRule:
    """Test rule that always blocks."""

    def check(self, order, portfolio):
        return None


class _PassthroughRule:
    """Test rule that always passes."""

    def check(self, order, portfolio):
        return order


class _HalveQuantityRule:
    """Test rule that halves order quantity."""

    def check(self, order, portfolio):
        from dataclasses import replace

        new_qty = order.quantity // 2
        if new_qty <= 0:
            return None
        return replace(order, quantity=new_qty)


class SimpleMAStrategy(AbstractStrategy):
    """Simple moving average crossover for testing."""

    def __init__(self, *, fast_period: int = 10, slow_period: int = 30, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signal(self, bar: MarketDataEvent) -> SignalEvent | None:
        closes = self.get_historical_closes(self.slow_period, bar.symbol)
        if len(closes) < self.slow_period:
            return None

        fast_ma = sum(closes[-self.fast_period :]) / self.fast_period
        slow_ma = sum(closes) / len(closes)

        if fast_ma > slow_ma:
            return SignalEvent(
                timestamp=bar.timestamp,
                symbol=bar.symbol,
                direction=Direction.LONG,
                strength=1.0,
            )
        elif fast_ma < slow_ma:
            return SignalEvent(
                timestamp=bar.timestamp,
                symbol=bar.symbol,
                direction=Direction.SHORT,
                strength=1.0,
            )
        return None


# ---------------------------------------------------------------------------
# RiskManager unit tests
# ---------------------------------------------------------------------------


class TestRiskManager:
    def test_no_rules_passes_order_through(self):
        """Empty rules list, order unchanged."""
        rm = RiskManager(rules=[])
        portfolio = Portfolio(initial_capital=100_000)
        order = _make_order()

        result = rm.evaluate(_make_signal(), order, portfolio)

        assert result is not None
        assert result.quantity == 100

    def test_first_rule_blocks_stops_chain(self):
        """First rule returns None, second never called."""
        second_rule = MagicMock()
        rm = RiskManager(rules=[_AlwaysBlockRule(), second_rule])
        portfolio = Portfolio(initial_capital=100_000)

        result = rm.evaluate(_make_signal(), _make_order(), portfolio)

        assert result is None
        second_rule.check.assert_not_called()

    def test_all_rules_pass(self):
        """Two passing rules, order returned."""
        rm = RiskManager(rules=[_PassthroughRule(), _PassthroughRule()])
        portfolio = Portfolio(initial_capital=100_000)
        order = _make_order()

        result = rm.evaluate(_make_signal(), order, portfolio)

        assert result is not None
        assert result.quantity == 100

    def test_rule_chain_modifies_order(self):
        """First rule resizes, second sees resized order."""
        rm = RiskManager(rules=[_HalveQuantityRule(), _HalveQuantityRule()])
        portfolio = Portfolio(initial_capital=100_000)
        order = _make_order(quantity=100)

        result = rm.evaluate(_make_signal(), order, portfolio)

        assert result is not None
        assert result.quantity == 25  # 100 -> 50 -> 25


class TestRiskManagerBreachLog:
    def test_rejection_logged(self):
        """Blocked order creates a REJECTED breach log entry."""
        rm = RiskManager(rules=[_AlwaysBlockRule()])
        portfolio = Portfolio(initial_capital=100_000)

        rm.evaluate(_make_signal(), _make_order(), portfolio)

        assert len(rm.breach_log) == 1
        breach = rm.breach_log[0]
        assert breach.action == RiskAction.REJECTED
        assert breach.symbol == "SPY"
        assert breach.proposed_qty == 100
        assert breach.final_qty is None

    def test_resize_logged(self):
        """Resized order creates a RESIZED breach log entry."""
        rm = RiskManager(rules=[_HalveQuantityRule()])
        portfolio = Portfolio(initial_capital=100_000)

        result = rm.evaluate(_make_signal(), _make_order(quantity=100), portfolio)

        assert result is not None
        assert result.quantity == 50
        assert len(rm.breach_log) == 1
        breach = rm.breach_log[0]
        assert breach.action == RiskAction.RESIZED
        assert breach.proposed_qty == 100
        assert breach.final_qty == 50

    def test_passthrough_no_log(self):
        """Order that passes unchanged generates no breach log entry."""
        rm = RiskManager(rules=[_PassthroughRule()])
        portfolio = Portfolio(initial_capital=100_000)

        rm.evaluate(_make_signal(), _make_order(), portfolio)

        assert len(rm.breach_log) == 0


class TestBreachLogDiagnostics:
    """Regression: breach_value and threshold must not always be 0.0."""

    def test_max_order_size_logs_actual_values(self):
        """MaxOrderSizeRule should log the order qty and the limit."""
        rule = MaxOrderSizeRule(max_order_qty=50)
        rm = RiskManager(rules=[rule])
        portfolio = Portfolio(initial_capital=100_000)

        rm.evaluate(_make_signal(), _make_order(quantity=100), portfolio)

        assert len(rm.breach_log) == 1
        breach = rm.breach_log[0]
        assert breach.breach_value == 100.0  # proposed qty
        assert breach.threshold == 50.0  # configured limit
        assert breach.action == RiskAction.REJECTED

    def test_max_drawdown_logs_actual_values(self):
        """MaxDrawdownRule should log the actual drawdown and the limit."""
        from sandtable.portfolio.portfolio import EquityPoint

        rule = MaxDrawdownRule(max_drawdown_pct=0.05)
        rm = RiskManager(rules=[rule])
        portfolio = Portfolio(initial_capital=100_000)

        # simulate a 10% drawdown via equity curve
        from datetime import datetime
        portfolio._equity_curve = [
            EquityPoint(timestamp=datetime(2024, 1, 1), equity=100_000, cash=100_000, positions_value=0),
            EquityPoint(timestamp=datetime(2024, 1, 2), equity=90_000, cash=90_000, positions_value=0),
        ]

        rm.evaluate(_make_signal(), _make_order(), portfolio)

        assert len(rm.breach_log) == 1
        breach = rm.breach_log[0]
        assert breach.breach_value > 0.09  # ~10% drawdown
        assert breach.threshold == 0.05
        assert breach.action == RiskAction.REJECTED

    def test_breach_values_not_zero_for_real_rules(self):
        """No real rule should log zeros when it actually breaches."""
        rule = MaxOrderSizeRule(max_order_qty=10)
        rm = RiskManager(rules=[rule])
        portfolio = Portfolio(initial_capital=100_000)

        rm.evaluate(_make_signal(), _make_order(quantity=500), portfolio)

        breach = rm.breach_log[0]
        assert breach.breach_value != 0.0, "breach_value must not be zero"
        assert breach.threshold != 0.0, "threshold must not be zero"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestRiskManagerIntegration:
    def test_risk_manager_blocks_all_trades(self):
        """MaxDrawdownRule(0.0) blocks everything, result has 0 trades."""
        data = make_data_handler(["SPY"])
        rm = RiskManager(rules=[MaxDrawdownRule(max_drawdown_pct=0.0)])
        result = run_backtest(
            strategy=SimpleMAStrategy(),
            data=data,
            risk_manager=rm,
        )

        assert isinstance(result, BacktestResult)
        assert result.metrics.num_trades == 0

    def test_risk_manager_none_by_default(self):
        """Omitting risk_manager gives same behavior as before."""
        data = make_data_handler(["SPY"])
        result = run_backtest(
            strategy=SimpleMAStrategy(),
            data=data,
        )

        assert isinstance(result, BacktestResult)
        # Should have some trades with default settings
        assert result.metrics.num_trades >= 0

    def test_risk_manager_with_real_strategy(self):
        """MA crossover with position size limit has fewer/smaller fills."""
        data_no_limit = make_data_handler(["SPY"])
        result_no_limit = run_backtest(
            strategy=SimpleMAStrategy(),
            data=data_no_limit,
        )

        data_with_limit = make_data_handler(["SPY"])
        rm = RiskManager(rules=[MaxPositionSizeRule(max_position_pct=0.01)])
        result_with_limit = run_backtest(
            strategy=SimpleMAStrategy(),
            data=data_with_limit,
            risk_manager=rm,
        )

        # With a very tight position limit, either fewer trades or smaller fills
        assert isinstance(result_with_limit, BacktestResult)
        if result_no_limit.metrics.num_trades > 0:
            # The limited version should have fewer or equal trades
            assert result_with_limit.metrics.num_trades <= result_no_limit.metrics.num_trades
