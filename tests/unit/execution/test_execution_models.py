"""
tests/execution/test_execution_models.py

Tests for execution models (slippage, impact, simulator).
"""

from datetime import datetime

import pytest

from sandtable.core.events import Direction, MarketDataEvent, OrderEvent, OrderType
from sandtable.execution.impact import NoMarketImpact, SquareRootImpactModel
from sandtable.execution.simulator import ExecutionConfig, ExecutionSimulator
from sandtable.execution.slippage import FixedSlippage, SpreadSlippage, ZeroSlippage


@pytest.fixture
def sample_bar() -> MarketDataEvent:
    """
    Create a sample market data bar.
    """
    return MarketDataEvent(
        timestamp=datetime(2024, 1, 15),
        symbol="SPY",
        open=450.0,
        high=455.0,
        low=448.0,
        close=452.0,
        volume=80_000_000,
    )


@pytest.fixture
def buy_order() -> OrderEvent:
    """
    Create a sample buy order.
    """
    return OrderEvent(
        timestamp=datetime(2024, 1, 15),
        symbol="SPY",
        direction=Direction.LONG,
        quantity=1000,
        order_type=OrderType.MARKET,
    )


@pytest.fixture
def sell_order() -> OrderEvent:
    """
    Create a sample sell order.
    """
    return OrderEvent(
        timestamp=datetime(2024, 1, 15),
        symbol="SPY",
        direction=Direction.SHORT,
        quantity=1000,
        order_type=OrderType.MARKET,
    )


class TestZeroSlippage:
    """
    Test ZeroSlippage model.
    """

    def test_returns_zero(self, buy_order: OrderEvent, sample_bar: MarketDataEvent):
        """
        ZeroSlippage should return 0.
        """
        model = ZeroSlippage()
        slippage = model.calculate_slippage(buy_order, sample_bar, sample_bar.close)
        assert slippage == 0.0


class TestFixedSlippage:
    """
    Test FixedSlippage model.
    """

    def test_correct_bps_calculation(
        self, buy_order: OrderEvent, sample_bar: MarketDataEvent
    ):
        """
        FixedSlippage should calculate correct basis points.
        """
        model = FixedSlippage(bps=10)  # 10 basis points = 0.1%
        slippage = model.calculate_slippage(buy_order, sample_bar, sample_bar.close)

        expected = sample_bar.close * (10 / 10000)  # 452 * 0.001 = 0.452
        assert slippage == pytest.approx(expected)

    def test_different_bps_values(
        self, buy_order: OrderEvent, sample_bar: MarketDataEvent
    ):
        """
        Different bps values should produce proportional slippage.
        """
        model5 = FixedSlippage(bps=5)
        model10 = FixedSlippage(bps=10)

        slip5 = model5.calculate_slippage(buy_order, sample_bar, sample_bar.close)
        slip10 = model10.calculate_slippage(buy_order, sample_bar, sample_bar.close)

        assert slip10 == pytest.approx(slip5 * 2)


class TestSpreadSlippage:
    """
    Test SpreadSlippage model.
    """

    def test_uses_bar_range(self, buy_order: OrderEvent, sample_bar: MarketDataEvent):
        """
        SpreadSlippage should use bar's high-low range.
        """
        model = SpreadSlippage(spread_fraction=0.5)
        slippage = model.calculate_slippage(buy_order, sample_bar, sample_bar.close)

        bar_range = sample_bar.high - sample_bar.low  # 455 - 448 = 7
        expected = bar_range * 0.5 * 0.5  # half spread = 1.75
        assert slippage == pytest.approx(expected)


class TestNoMarketImpact:
    """
    Test NoMarketImpact model.
    """

    def test_returns_zero(self, buy_order: OrderEvent, sample_bar: MarketDataEvent):
        """
        NoMarketImpact should return 0.
        """
        model = NoMarketImpact()
        impact = model.calculate_impact(buy_order, sample_bar, sample_bar.close)
        assert impact == 0.0


class TestSquareRootImpactModel:
    """
    Test SquareRootImpactModel.
    """

    def test_impact_increases_with_size(self, sample_bar: MarketDataEvent):
        """
        Larger orders should have more impact.
        """
        model = SquareRootImpactModel(eta=0.1, sigma=0.02)

        small_order = OrderEvent(
            timestamp=datetime(2024, 1, 15),
            symbol="SPY",
            direction=Direction.LONG,
            quantity=100,
            order_type=OrderType.MARKET,
        )

        large_order = OrderEvent(
            timestamp=datetime(2024, 1, 15),
            symbol="SPY",
            direction=Direction.LONG,
            quantity=10000,
            order_type=OrderType.MARKET,
        )

        small_impact = model.calculate_impact(small_order, sample_bar, sample_bar.close)
        large_impact = model.calculate_impact(large_order, sample_bar, sample_bar.close)

        assert large_impact > small_impact

    def test_square_root_relationship(self, sample_bar: MarketDataEvent):
        """
        Impact should scale with sqrt of order size.
        """
        model = SquareRootImpactModel(eta=0.1, sigma=0.02)

        order1 = OrderEvent(
            timestamp=datetime(2024, 1, 15),
            symbol="SPY",
            direction=Direction.LONG,
            quantity=1000,
            order_type=OrderType.MARKET,
        )

        order4 = OrderEvent(
            timestamp=datetime(2024, 1, 15),
            symbol="SPY",
            direction=Direction.LONG,
            quantity=4000,  # 4x the size
            order_type=OrderType.MARKET,
        )

        impact1 = model.calculate_impact(order1, sample_bar, sample_bar.close)
        impact4 = model.calculate_impact(order4, sample_bar, sample_bar.close)

        # sqrt(4) = 2, so impact4 should be ~2x impact1
        assert impact4 == pytest.approx(impact1 * 2, rel=0.01)


class TestExecutionSimulator:
    """
    Test ExecutionSimulator.
    """

    def test_buy_fills_higher(
        self, buy_order: OrderEvent, sample_bar: MarketDataEvent
    ):
        """
        Buy orders should fill at or above close (with costs).
        """
        simulator = ExecutionSimulator(
            slippage_model=FixedSlippage(bps=10),
            impact_model=NoMarketImpact(),
        )

        fill = simulator.process_order(buy_order, sample_bar)

        assert fill.fill_price > sample_bar.close

    def test_sell_fills_lower(
        self, sell_order: OrderEvent, sample_bar: MarketDataEvent
    ):
        """
        Sell orders should fill at or below close (with costs).
        """
        simulator = ExecutionSimulator(
            slippage_model=FixedSlippage(bps=10),
            impact_model=NoMarketImpact(),
        )

        fill = simulator.process_order(sell_order, sample_bar)

        assert fill.fill_price < sample_bar.close

    def test_fill_clamped_to_bar_range(self, sample_bar: MarketDataEvent):
        """
        Fill price should be clamped to bar's [low, high].
        """
        # Create huge order with high impact
        huge_order = OrderEvent(
            timestamp=datetime(2024, 1, 15),
            symbol="SPY",
            direction=Direction.LONG,
            quantity=50_000_000,
            order_type=OrderType.MARKET,
        )

        simulator = ExecutionSimulator(
            slippage_model=FixedSlippage(bps=100),  # 1%
            impact_model=SquareRootImpactModel(eta=1.0),
        )

        fill = simulator.process_order(huge_order, sample_bar)

        # Should be clamped to high
        assert fill.fill_price <= sample_bar.high
        assert fill.fill_price >= sample_bar.low

    def test_commission_minimum_enforced(self, sample_bar: MarketDataEvent):
        """
        Commission minimum should be enforced.
        """
        tiny_order = OrderEvent(
            timestamp=datetime(2024, 1, 15),
            symbol="SPY",
            direction=Direction.LONG,
            quantity=1,  # Very small order
            order_type=OrderType.MARKET,
        )

        config = ExecutionConfig(
            commission_per_share=0.005,
            commission_minimum=1.0,  # Minimum $1
        )

        simulator = ExecutionSimulator(config=config)
        fill = simulator.process_order(tiny_order, sample_bar)

        # Commission should be at least minimum
        assert fill.commission >= 1.0

    def test_commission_per_share(self, buy_order: OrderEvent, sample_bar: MarketDataEvent):
        """
        Commission should scale with order size when above minimum.
        """
        config = ExecutionConfig(
            commission_per_share=0.01,
            commission_minimum=0.0,
        )

        simulator = ExecutionSimulator(config=config)
        fill = simulator.process_order(buy_order, sample_bar)

        expected_commission = buy_order.quantity * 0.01  # 1000 * 0.01 = 10
        assert fill.commission == pytest.approx(expected_commission)

    def test_fill_event_has_correct_fields(
        self, buy_order: OrderEvent, sample_bar: MarketDataEvent
    ):
        """
        Fill event should have all expected fields.
        """
        simulator = ExecutionSimulator(
            slippage_model=FixedSlippage(bps=5),
            impact_model=SquareRootImpactModel(eta=0.1),
        )

        fill = simulator.process_order(buy_order, sample_bar)

        assert fill.timestamp == buy_order.timestamp
        assert fill.symbol == buy_order.symbol
        assert fill.direction == buy_order.direction
        assert fill.quantity == buy_order.quantity
        assert fill.fill_price > 0
        assert fill.commission > 0
        assert fill.slippage >= 0
        assert fill.market_impact >= 0

    def test_zero_costs_baseline(
        self, buy_order: OrderEvent, sample_bar: MarketDataEvent
    ):
        """
        With zero costs, should fill at close.
        """
        config = ExecutionConfig(
            commission_per_share=0.0,
            commission_minimum=0.0,
        )

        simulator = ExecutionSimulator(
            config=config,
            slippage_model=ZeroSlippage(),
            impact_model=NoMarketImpact(),
        )

        fill = simulator.process_order(buy_order, sample_bar)

        assert fill.fill_price == sample_bar.close
        assert fill.slippage == 0.0
        assert fill.market_impact == 0.0
