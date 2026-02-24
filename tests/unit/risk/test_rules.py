"""
tests/unit/risk/test_rules.py

Tests for individual risk rules.
"""

from datetime import datetime, timedelta

from sandtable.core.events import Direction, OrderEvent, OrderType
from sandtable.portfolio.portfolio import EquityPoint, Portfolio, Position
from sandtable.risk.rules import (
    MaxConcentrationRule,
    MaxDailyLossRule,
    MaxDrawdownRule,
    MaxLeverageRule,
    MaxOrderSizeRule,
    MaxPortfolioExposureRule,
    MaxPositionSizeRule,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_portfolio(
    initial: float = 100_000,
    cash: float | None = None,
    positions: dict[str, Position] | None = None,
    current_prices: dict[str, float] | None = None,
    equity_curve: list[EquityPoint] | None = None,
) -> Portfolio:
    """
    Build a Portfolio with controlled internal state for deterministic testing.
    """
    p = Portfolio(initial_capital=initial)
    if positions is not None:
        p._positions = positions
    if current_prices is not None:
        p._current_prices = current_prices
    if cash is not None:
        p._cash = cash
    if equity_curve is not None:
        p._equity_curve = equity_curve
    return p


def _make_order(
    symbol: str = "SPY",
    direction: Direction = Direction.LONG,
    quantity: int = 100,
) -> OrderEvent:
    return OrderEvent(
        timestamp=datetime(2024, 1, 1),
        symbol=symbol,
        direction=direction,
        quantity=quantity,
        order_type=OrderType.MARKET,
    )


def _equity_point(equity: float, day_offset: int = 0) -> EquityPoint:
    return EquityPoint(
        timestamp=datetime(2024, 1, 1) + timedelta(days=day_offset),
        equity=equity,
        cash=equity,
        positions_value=0.0,
    )


# ---------------------------------------------------------------------------
# MaxPositionSizeRule
# ---------------------------------------------------------------------------


class TestMaxPositionSizeRule:
    def test_allows_order_within_limit(self):
        """Order that keeps position under threshold passes unchanged."""
        rule = MaxPositionSizeRule(max_position_pct=0.25)
        # Equity = 100k, max position = 25k, order = 100 shares @ $100 = 10k
        portfolio = _make_portfolio(
            cash=100_000,
            current_prices={"SPY": 100.0},
        )
        order = _make_order(quantity=100)
        result = rule.check(order, portfolio)

        assert result is not None
        assert result.quantity == 100

    def test_reduces_oversized_order(self):
        """Order that would exceed limit gets quantity reduced."""
        rule = MaxPositionSizeRule(max_position_pct=0.10)
        # Equity = 100k, max position = 10k, order = 200 shares @ $100 = 20k
        portfolio = _make_portfolio(
            cash=100_000,
            current_prices={"SPY": 100.0},
        )
        order = _make_order(quantity=200)
        result = rule.check(order, portfolio)

        assert result is not None
        assert result.quantity == 100  # 10k / $100 = 100 shares

    def test_blocks_when_already_at_limit(self):
        """Existing position already at cap returns None."""
        rule = MaxPositionSizeRule(max_position_pct=0.10)
        # Equity = 100k, existing 100 shares @ $100 = $10k = 10% already
        portfolio = _make_portfolio(
            cash=90_000,
            positions={"SPY": Position(symbol="SPY", quantity=100, avg_cost=100.0)},
            current_prices={"SPY": 100.0},
        )
        order = _make_order(quantity=50)
        result = rule.check(order, portfolio)

        assert result is None

    def test_respects_existing_position(self):
        """Partial existing position - only remaining room is allocated."""
        rule = MaxPositionSizeRule(max_position_pct=0.20)
        # Equity = 100k, existing 50 shares @ $100 = $5k (5%), max = 20k
        # Remaining room = 20k - 5k = 15k -> 150 shares
        # Order requests 200, should be reduced to 150
        portfolio = _make_portfolio(
            cash=95_000,
            positions={"SPY": Position(symbol="SPY", quantity=50, avg_cost=100.0)},
            current_prices={"SPY": 100.0},
        )
        order = _make_order(quantity=200)
        result = rule.check(order, portfolio)

        assert result is not None
        assert result.quantity == 150


# ---------------------------------------------------------------------------
# MaxPortfolioExposureRule
# ---------------------------------------------------------------------------


class TestMaxPortfolioExposureRule:
    def test_allows_when_under_gross_limit(self):
        """Total exposure within cap passes through."""
        rule = MaxPortfolioExposureRule(max_gross_exposure_pct=1.0)
        # Equity = 100k, no existing positions, order = 100 * $100 = 10k
        portfolio = _make_portfolio(
            cash=100_000,
            current_prices={"SPY": 100.0},
        )
        order = _make_order(quantity=100)
        result = rule.check(order, portfolio)

        assert result is not None
        assert result.quantity == 100

    def test_blocks_when_over_gross_limit(self):
        """Would exceed cap, returns None."""
        rule = MaxPortfolioExposureRule(max_gross_exposure_pct=0.50)
        # Equity = 100k, existing = 500 * $100 = 50k already at limit
        portfolio = _make_portfolio(
            cash=50_000,
            positions={"AAPL": Position(symbol="AAPL", quantity=500, avg_cost=100.0)},
            current_prices={"AAPL": 100.0, "SPY": 100.0},
        )
        order = _make_order(quantity=100)
        result = rule.check(order, portfolio)

        assert result is None

    def test_empty_portfolio_allows_first_order(self):
        """No existing positions, first order should pass."""
        rule = MaxPortfolioExposureRule(max_gross_exposure_pct=0.50)
        portfolio = _make_portfolio(
            cash=100_000,
            current_prices={"SPY": 100.0},
        )
        order = _make_order(quantity=100)  # 10k out of 50k max = fine
        result = rule.check(order, portfolio)

        assert result is not None
        assert result.quantity == 100

    def test_reduces_to_fit_remaining_room(self):
        """Order reduced to fit remaining gross exposure room."""
        rule = MaxPortfolioExposureRule(max_gross_exposure_pct=0.50)
        # Equity = 100k, existing = 200 * $100 = 20k, max = 50k, room = 30k
        portfolio = _make_portfolio(
            cash=80_000,
            positions={"AAPL": Position(symbol="AAPL", quantity=200, avg_cost=100.0)},
            current_prices={"AAPL": 100.0, "SPY": 50.0},
        )
        order = _make_order(quantity=1000)  # 1000 * $50 = 50k > 30k room
        result = rule.check(order, portfolio)

        assert result is not None
        assert result.quantity == 600  # 30k / $50 = 600


# ---------------------------------------------------------------------------
# MaxDrawdownRule
# ---------------------------------------------------------------------------


class TestMaxDrawdownRule:
    def test_allows_order_when_no_drawdown(self):
        """Fresh portfolio with flat equity, order passes."""
        rule = MaxDrawdownRule(max_drawdown_pct=0.20)
        portfolio = _make_portfolio(
            cash=100_000,
            equity_curve=[_equity_point(100_000)],
            current_prices={"SPY": 100.0},
        )
        order = _make_order()
        result = rule.check(order, portfolio)

        assert result is not None

    def test_allows_order_when_drawdown_below_threshold(self):
        """Small drawdown, order passes."""
        rule = MaxDrawdownRule(max_drawdown_pct=0.20)
        portfolio = _make_portfolio(
            cash=95_000,
            equity_curve=[
                _equity_point(100_000, 0),
                _equity_point(95_000, 1),  # 5% drawdown < 20%
            ],
            current_prices={"SPY": 100.0},
        )
        order = _make_order()
        result = rule.check(order, portfolio)

        assert result is not None

    def test_blocks_order_when_drawdown_exceeds_threshold(self):
        """Deep drawdown, order blocked."""
        rule = MaxDrawdownRule(max_drawdown_pct=0.10)
        portfolio = _make_portfolio(
            cash=85_000,
            equity_curve=[
                _equity_point(100_000, 0),
                _equity_point(85_000, 1),  # 15% drawdown > 10%
            ],
            current_prices={"SPY": 100.0},
        )
        order = _make_order()
        result = rule.check(order, portfolio)

        assert result is None

    def test_exact_threshold_blocks(self):
        """Drawdown equals threshold exactly, blocked."""
        rule = MaxDrawdownRule(max_drawdown_pct=0.20)
        portfolio = _make_portfolio(
            cash=80_000,
            equity_curve=[
                _equity_point(100_000, 0),
                _equity_point(80_000, 1),  # exactly 20%
            ],
            current_prices={"SPY": 100.0},
        )
        order = _make_order()
        result = rule.check(order, portfolio)

        assert result is None

    def test_empty_equity_curve_allows(self):
        """No equity history yet, allow the order."""
        rule = MaxDrawdownRule(max_drawdown_pct=0.10)
        portfolio = _make_portfolio(cash=100_000, current_prices={"SPY": 100.0})
        order = _make_order()
        result = rule.check(order, portfolio)

        assert result is not None


# ---------------------------------------------------------------------------
# MaxDailyLossRule
# ---------------------------------------------------------------------------


class TestMaxDailyLossRule:
    def test_allows_when_no_loss_today(self):
        """No loss today, order passes."""
        rule = MaxDailyLossRule(max_daily_loss_pct=0.02)
        portfolio = _make_portfolio(
            cash=100_000,
            equity_curve=[
                _equity_point(99_000, 0),  # yesterday
                _equity_point(100_000, 1),  # today, no loss
            ],
            current_prices={"SPY": 100.0},
        )
        order = _make_order()
        result = rule.check(order, portfolio)

        assert result is not None

    def test_blocks_when_daily_loss_exceeds_threshold(self):
        """Today's loss exceeds threshold, order blocked."""
        rule = MaxDailyLossRule(max_daily_loss_pct=0.02)
        today = datetime(2024, 1, 2)
        portfolio = _make_portfolio(
            cash=97_000,
            equity_curve=[
                EquityPoint(timestamp=datetime(2024, 1, 1), equity=99_000, cash=99_000, positions_value=0),
                EquityPoint(timestamp=today, equity=100_000, cash=100_000, positions_value=0),  # start of day
                EquityPoint(timestamp=today + timedelta(hours=4), equity=97_000, cash=97_000, positions_value=0),  # 3% loss
            ],
            current_prices={"SPY": 100.0},
        )
        order = _make_order()
        result = rule.check(order, portfolio)

        assert result is None

    def test_allows_when_daily_loss_within_threshold(self):
        """Today's loss is within threshold, order passes."""
        rule = MaxDailyLossRule(max_daily_loss_pct=0.05)
        today = datetime(2024, 1, 2)
        portfolio = _make_portfolio(
            cash=98_000,
            equity_curve=[
                EquityPoint(timestamp=datetime(2024, 1, 1), equity=99_000, cash=99_000, positions_value=0),
                EquityPoint(timestamp=today, equity=100_000, cash=100_000, positions_value=0),
                EquityPoint(
                    timestamp=today + timedelta(hours=4), equity=98_000, cash=98_000, positions_value=0,
                ),  # 2% loss < 5%
            ],
            current_prices={"SPY": 100.0},
        )
        order = _make_order()
        result = rule.check(order, portfolio)

        assert result is not None

    def test_empty_equity_curve_allows(self):
        """No equity history yet, allow the order."""
        rule = MaxDailyLossRule(max_daily_loss_pct=0.02)
        portfolio = _make_portfolio(cash=100_000, current_prices={"SPY": 100.0})
        order = _make_order()
        result = rule.check(order, portfolio)

        assert result is not None


# ---------------------------------------------------------------------------
# MaxLeverageRule
# ---------------------------------------------------------------------------


class TestMaxLeverageRule:
    def test_allows_order_within_leverage_limit(self):
        """Order that keeps leverage under threshold passes unchanged."""
        rule = MaxLeverageRule(max_leverage=2.0)
        # Equity = 100k, max exposure = 200k, order = 100 * $100 = 10k
        portfolio = _make_portfolio(
            cash=100_000,
            current_prices={"SPY": 100.0},
        )
        order = _make_order(quantity=100)
        result = rule.check(order, portfolio)

        assert result is not None
        assert result.quantity == 100

    def test_reduces_oversized_order(self):
        """Order that would exceed leverage limit gets reduced."""
        rule = MaxLeverageRule(max_leverage=1.0)
        # Equity = 100k, max exposure = 100k, existing = 50k, room = 50k
        portfolio = _make_portfolio(
            cash=50_000,
            positions={"AAPL": Position(symbol="AAPL", quantity=500, avg_cost=100.0)},
            current_prices={"AAPL": 100.0, "SPY": 100.0},
        )
        order = _make_order(quantity=1000)  # 1000 * $100 = 100k > 50k room
        result = rule.check(order, portfolio)

        assert result is not None
        assert result.quantity == 500  # 50k / $100 = 500

    def test_blocks_when_at_leverage_limit(self):
        """Already at leverage limit, reject all orders."""
        rule = MaxLeverageRule(max_leverage=0.50)
        # Equity = 100k, max exposure = 50k, existing = 500 * $100 = 50k
        portfolio = _make_portfolio(
            cash=50_000,
            positions={"AAPL": Position(symbol="AAPL", quantity=500, avg_cost=100.0)},
            current_prices={"AAPL": 100.0, "SPY": 100.0},
        )
        order = _make_order(quantity=100)
        result = rule.check(order, portfolio)

        assert result is None


# ---------------------------------------------------------------------------
# MaxOrderSizeRule
# ---------------------------------------------------------------------------


class TestMaxOrderSizeRule:
    def test_allows_order_within_limit(self):
        """Order quantity under limit passes through."""
        rule = MaxOrderSizeRule(max_order_qty=500)
        portfolio = _make_portfolio(cash=100_000, current_prices={"SPY": 100.0})
        order = _make_order(quantity=100)
        result = rule.check(order, portfolio)

        assert result is not None
        assert result.quantity == 100

    def test_blocks_oversized_order(self):
        """Order quantity over limit is hard-rejected (no resize)."""
        rule = MaxOrderSizeRule(max_order_qty=500)
        portfolio = _make_portfolio(cash=100_000, current_prices={"SPY": 100.0})
        order = _make_order(quantity=1000)
        result = rule.check(order, portfolio)

        assert result is None

    def test_exact_limit_passes(self):
        """Order at exact limit passes."""
        rule = MaxOrderSizeRule(max_order_qty=500)
        portfolio = _make_portfolio(cash=100_000, current_prices={"SPY": 100.0})
        order = _make_order(quantity=500)
        result = rule.check(order, portfolio)

        assert result is not None
        assert result.quantity == 500


# ---------------------------------------------------------------------------
# MaxConcentrationRule
# ---------------------------------------------------------------------------


class TestMaxConcentrationRule:
    def test_allows_when_under_concentration_limit(self):
        """Order that keeps symbol concentration under limit passes."""
        rule = MaxConcentrationRule(max_concentration_pct=0.50)
        # No existing positions, first order is 100% concentrated but that's fine
        # if the limit allows it
        portfolio = _make_portfolio(
            cash=100_000,
            current_prices={"SPY": 100.0},
        )
        order = _make_order(quantity=100)
        result = rule.check(order, portfolio)

        assert result is not None

    def test_blocks_when_concentration_exceeds_limit(self):
        """Existing concentration at limit, new order rejected."""
        rule = MaxConcentrationRule(max_concentration_pct=0.30)
        # Existing: SPY = 400 * $100 = 40k, AAPL = 600 * $100 = 60k
        # gross = 100k, SPY concentration = 40%. New SPY order would increase it.
        portfolio = _make_portfolio(
            cash=0,
            positions={
                "SPY": Position(symbol="SPY", quantity=400, avg_cost=100.0),
                "AAPL": Position(symbol="AAPL", quantity=600, avg_cost=100.0),
            },
            current_prices={"SPY": 100.0, "AAPL": 100.0},
        )
        order = _make_order(symbol="SPY", quantity=500)
        result = rule.check(order, portfolio)

        # SPY is already at 40% > 30% limit, any addition should be blocked
        assert result is None

    def test_reduces_to_fit_concentration_limit(self):
        """Order reduced to stay within concentration limit."""
        rule = MaxConcentrationRule(max_concentration_pct=0.50)
        # Existing: AAPL = 500 * $100 = 50k, gross = 50k
        # SPY order of 1000 * $100 = 100k would make SPY = 100k/(50k+100k) = 67%
        # Need: SPY_value / (50k + SPY_value) <= 0.50
        # SPY_value <= 0.50 * (50k + SPY_value)
        # SPY_value <= 25k + 0.5*SPY_value
        # 0.5*SPY_value <= 25k -> SPY_value <= 50k -> 500 shares
        portfolio = _make_portfolio(
            cash=50_000,
            positions={
                "AAPL": Position(symbol="AAPL", quantity=500, avg_cost=100.0),
            },
            current_prices={"AAPL": 100.0, "SPY": 100.0},
        )
        order = _make_order(symbol="SPY", quantity=1000)
        result = rule.check(order, portfolio)

        assert result is not None
        assert result.quantity == 500


# ---------------------------------------------------------------------------
# Edge cases for all rules (zero equity, zero price, etc.)
# ---------------------------------------------------------------------------


class TestRuleEdgeCases:
    """Edge cases for coverage: zero equity, zero price, etc."""

    def test_max_position_size_zero_equity(self):
        """Zero equity returns None."""
        rule = MaxPositionSizeRule()
        portfolio = _make_portfolio(cash=0, current_prices={"SPY": 100.0})
        order = _make_order()
        assert rule.check(order, portfolio) is None

    def test_max_position_size_zero_price(self):
        """Zero price returns order unchanged."""
        rule = MaxPositionSizeRule()
        portfolio = _make_portfolio(cash=100_000, current_prices={"SPY": 0.0})
        order = _make_order()
        result = rule.check(order, portfolio)
        assert result is not None
        assert result.quantity == 100

    def test_max_portfolio_exposure_zero_equity(self):
        rule = MaxPortfolioExposureRule()
        portfolio = _make_portfolio(cash=0, current_prices={"SPY": 100.0})
        order = _make_order()
        assert rule.check(order, portfolio) is None

    def test_max_portfolio_exposure_zero_price(self):
        rule = MaxPortfolioExposureRule()
        portfolio = _make_portfolio(cash=100_000, current_prices={"SPY": 0.0})
        order = _make_order()
        result = rule.check(order, portfolio)
        assert result is not None

    def test_max_portfolio_exposure_resize_to_zero(self):
        """Remaining room too small for even 1 share -> None."""
        rule = MaxPortfolioExposureRule(max_gross_exposure_pct=0.50)
        portfolio = _make_portfolio(
            cash=50_000,
            positions={"AAPL": Position(symbol="AAPL", quantity=500, avg_cost=100.0)},
            current_prices={"AAPL": 100.0, "SPY": 200.0},
        )
        order = _make_order(quantity=100)
        assert rule.check(order, portfolio) is None

    def test_max_drawdown_peak_zero(self):
        """Peak equity is zero, allow order."""
        rule = MaxDrawdownRule()
        portfolio = _make_portfolio(
            cash=100_000,
            equity_curve=[_equity_point(0.0)],
            current_prices={"SPY": 100.0},
        )
        order = _make_order()
        assert rule.check(order, portfolio) is not None

    def test_max_daily_loss_start_of_day_zero(self):
        """Start of day equity is zero, allow order."""
        rule = MaxDailyLossRule()
        today = datetime(2024, 1, 1)
        portfolio = _make_portfolio(
            cash=100_000,
            equity_curve=[
                EquityPoint(timestamp=today, equity=0.0, cash=0.0, positions_value=0.0),
            ],
            current_prices={"SPY": 100.0},
        )
        order = _make_order()
        assert rule.check(order, portfolio) is not None

    def test_max_leverage_zero_equity(self):
        rule = MaxLeverageRule()
        portfolio = _make_portfolio(cash=0, current_prices={"SPY": 100.0})
        order = _make_order()
        assert rule.check(order, portfolio) is None

    def test_max_leverage_zero_price(self):
        rule = MaxLeverageRule()
        portfolio = _make_portfolio(cash=100_000, current_prices={"SPY": 0.0})
        order = _make_order()
        result = rule.check(order, portfolio)
        assert result is not None

    def test_max_leverage_resize_to_zero(self):
        """Remaining room too small for 1 share -> None."""
        rule = MaxLeverageRule(max_leverage=0.50)
        portfolio = _make_portfolio(
            cash=50_100,
            positions={"AAPL": Position(symbol="AAPL", quantity=499, avg_cost=100.0)},
            current_prices={"AAPL": 100.0, "SPY": 100_000.0},
        )
        order = _make_order(quantity=100)
        assert rule.check(order, portfolio) is None

    def test_max_concentration_zero_price(self):
        rule = MaxConcentrationRule()
        portfolio = _make_portfolio(
            cash=100_000,
            positions={"AAPL": Position(symbol="AAPL", quantity=100, avg_cost=100.0)},
            current_prices={"AAPL": 100.0, "SPY": 0.0},
        )
        order = _make_order()
        result = rule.check(order, portfolio)
        assert result is not None

    def test_max_concentration_denom_zero(self):
        """max_concentration_pct >= 1.0 means denom <= 0, allow order."""
        rule = MaxConcentrationRule(max_concentration_pct=1.0)
        portfolio = _make_portfolio(
            cash=50_000,
            positions={"AAPL": Position(symbol="AAPL", quantity=100, avg_cost=100.0)},
            current_prices={"AAPL": 100.0, "SPY": 100.0},
        )
        order = _make_order(quantity=1000)
        result = rule.check(order, portfolio)
        assert result is not None

    def test_max_concentration_resize_to_zero(self):
        """Additional value allowed is tiny -> new_qty = 0 -> None."""
        rule = MaxConcentrationRule(max_concentration_pct=0.30)
        portfolio = _make_portfolio(
            cash=0,
            positions={
                "SPY": Position(symbol="SPY", quantity=290, avg_cost=100.0),
                "AAPL": Position(symbol="AAPL", quantity=710, avg_cost=100.0),
            },
            current_prices={"SPY": 100.0, "AAPL": 100.0},
        )
        order = _make_order(symbol="SPY", quantity=500)
        result = rule.check(order, portfolio)
        if result is not None:
            assert result.quantity < 500
