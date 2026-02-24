"""
tests/portfolio/test_portfolio.py

Tests for portfolio management.
"""

from datetime import datetime

import pytest

from sandtable.core.events import Direction, FillEvent, MarketDataEvent, SignalEvent
from sandtable.portfolio.portfolio import Portfolio, Position


@pytest.fixture
def portfolio() -> Portfolio:
    """Create a portfolio with default settings."""
    return Portfolio(initial_capital=100_000, position_size_pct=0.10)


@pytest.fixture
def sample_bar() -> MarketDataEvent:
    """Create a sample market data bar."""
    return MarketDataEvent(
        timestamp=datetime(2024, 1, 15),
        symbol="SPY",
        open=450.0,
        high=455.0,
        low=448.0,
        close=450.0,
        volume=80_000_000,
    )


def make_fill(
    direction: Direction,
    quantity: int,
    fill_price: float,
    symbol: str = "SPY",
) -> FillEvent:
    """Helper to create a FillEvent."""
    return FillEvent(
        timestamp=datetime(2024, 1, 15),
        symbol=symbol,
        direction=direction,
        quantity=quantity,
        fill_price=fill_price,
        commission=5.0,
        slippage=0.5,
        market_impact=0.1,
    )


class TestPosition:
    """Test Position dataclass."""

    def test_long_position(self):
        """Long position has positive quantity."""
        pos = Position(symbol="SPY", quantity=100, avg_cost=450.0)
        assert pos.is_long
        assert not pos.is_short
        assert not pos.is_flat

    def test_short_position(self):
        """Short position has negative quantity."""
        pos = Position(symbol="SPY", quantity=-100, avg_cost=450.0)
        assert not pos.is_long
        assert pos.is_short
        assert not pos.is_flat

    def test_flat_position(self):
        """Flat position has zero quantity."""
        pos = Position(symbol="SPY", quantity=0, avg_cost=0.0)
        assert not pos.is_long
        assert not pos.is_short
        assert pos.is_flat

    def test_market_value_long(self):
        """Market value for long position."""
        pos = Position(symbol="SPY", quantity=100, avg_cost=450.0)
        value = pos.market_value(460.0)
        assert value == 100 * 460.0

    def test_market_value_short(self):
        """Market value for short position (negative)."""
        pos = Position(symbol="SPY", quantity=-100, avg_cost=450.0)
        value = pos.market_value(460.0)
        assert value == -100 * 460.0

    def test_unrealized_pnl_long_profit(self):
        """Unrealized P&L for profitable long."""
        pos = Position(symbol="SPY", quantity=100, avg_cost=450.0)
        pnl = pos.unrealized_pnl(460.0)  # Price went up
        assert pnl == 100 * (460.0 - 450.0)  # +$1000

    def test_unrealized_pnl_long_loss(self):
        """Unrealized P&L for losing long."""
        pos = Position(symbol="SPY", quantity=100, avg_cost=450.0)
        pnl = pos.unrealized_pnl(440.0)  # Price went down
        assert pnl == 100 * (440.0 - 450.0)  # -$1000

    def test_unrealized_pnl_short_profit(self):
        """Unrealized P&L for profitable short."""
        pos = Position(symbol="SPY", quantity=-100, avg_cost=450.0)
        pnl = pos.unrealized_pnl(440.0)  # Price went down (good for short)
        assert pnl == -100 * (440.0 - 450.0)  # +$1000

    def test_unrealized_pnl_short_loss(self):
        """Unrealized P&L for losing short."""
        pos = Position(symbol="SPY", quantity=-100, avg_cost=450.0)
        pnl = pos.unrealized_pnl(460.0)  # Price went up (bad for short)
        assert pnl == -100 * (460.0 - 450.0)  # -$1000


class TestPortfolioBasics:
    """Test basic portfolio operations."""

    def test_initial_state(self, portfolio: Portfolio):
        """Portfolio should start with initial capital."""
        assert portfolio.cash == 100_000
        assert portfolio.equity() == 100_000
        assert len(portfolio.positions) == 0

    def test_update_price(self, portfolio: Portfolio):
        """Price updates should be stored."""
        portfolio.update_price("SPY", 450.0)
        # Internal state - check via market data
        portfolio.on_market_data(
            MarketDataEvent(
                datetime(2024, 1, 15), "SPY", 450, 455, 448, 451, 1000000
            )
        )
        # No direct way to check, but shouldn't error


class TestPortfolioBuyOperations:
    """Test buy (LONG) operations."""

    def test_buy_increases_position(self, portfolio: Portfolio):
        """Buying should increase position quantity."""
        fill = make_fill(Direction.LONG, 100, 450.0)
        portfolio.on_fill(fill)

        pos = portfolio.get_position("SPY")
        assert pos.quantity == 100

    def test_buy_decreases_cash(self, portfolio: Portfolio):
        """Buying should decrease cash by trade value + commission."""
        fill = make_fill(Direction.LONG, 100, 450.0)  # 100 * 450 = 45000, comm = 5
        portfolio.on_fill(fill)

        expected_cash = 100_000 - (100 * 450.0) - 5.0
        assert portfolio.cash == pytest.approx(expected_cash)

    def test_buy_sets_avg_cost(self, portfolio: Portfolio):
        """First buy should set average cost."""
        fill = make_fill(Direction.LONG, 100, 450.0)
        portfolio.on_fill(fill)

        pos = portfolio.get_position("SPY")
        assert pos.avg_cost == 450.0

    def test_multiple_buys_update_avg_cost(self, portfolio: Portfolio):
        """Multiple buys should update average cost."""
        fill1 = make_fill(Direction.LONG, 100, 450.0)
        fill2 = make_fill(Direction.LONG, 100, 460.0)

        portfolio.on_fill(fill1)
        portfolio.on_fill(fill2)

        pos = portfolio.get_position("SPY")
        assert pos.quantity == 200
        # Avg cost = (100*450 + 100*460) / 200 = 455
        assert pos.avg_cost == pytest.approx(455.0)


class TestPortfolioSellOperations:
    """Test sell (SHORT) operations."""

    def test_sell_decreases_position(self, portfolio: Portfolio):
        """Selling should decrease position quantity."""
        # First buy
        buy_fill = make_fill(Direction.LONG, 100, 450.0)
        portfolio.on_fill(buy_fill)

        # Then sell
        sell_fill = make_fill(Direction.SHORT, 50, 455.0)
        portfolio.on_fill(sell_fill)

        pos = portfolio.get_position("SPY")
        assert pos.quantity == 50

    def test_sell_increases_cash(self, portfolio: Portfolio):
        """Selling should increase cash."""
        # First buy
        buy_fill = make_fill(Direction.LONG, 100, 450.0)
        portfolio.on_fill(buy_fill)
        cash_after_buy = portfolio.cash

        # Then sell
        sell_fill = make_fill(Direction.SHORT, 50, 455.0)
        portfolio.on_fill(sell_fill)

        # Cash increases by (50 * 455) - 5 commission
        expected_cash = cash_after_buy + (50 * 455.0) - 5.0
        assert portfolio.cash == pytest.approx(expected_cash)

    def test_sell_realizes_pnl(self, portfolio: Portfolio):
        """Selling should realize P&L."""
        # Buy at 450
        buy_fill = make_fill(Direction.LONG, 100, 450.0)
        portfolio.on_fill(buy_fill)

        # Sell at 460 (profit)
        sell_fill = make_fill(Direction.SHORT, 100, 460.0)
        portfolio.on_fill(sell_fill)

        pos = portfolio.get_position("SPY")
        # Realized P&L = (460 - 450) * 100 = 1000
        assert pos.realized_pnl == pytest.approx(1000.0)


class TestPortfolioShortOperations:
    """Test short selling operations."""

    def test_short_creates_negative_position(self, portfolio: Portfolio):
        """Short selling creates negative position."""
        fill = make_fill(Direction.SHORT, 100, 450.0)
        portfolio.on_fill(fill)

        pos = portfolio.get_position("SPY")
        assert pos.quantity == -100
        assert pos.is_short

    def test_short_increases_cash(self, portfolio: Portfolio):
        """Short selling increases cash (receive proceeds)."""
        fill = make_fill(Direction.SHORT, 100, 450.0)  # Receive 45000 - 5 comm
        portfolio.on_fill(fill)

        expected_cash = 100_000 + (100 * 450.0) - 5.0
        assert portfolio.cash == pytest.approx(expected_cash)

    def test_cover_short_decreases_cash(self, portfolio: Portfolio):
        """Covering short decreases cash (pay to buy back)."""
        # Short sell
        short_fill = make_fill(Direction.SHORT, 100, 450.0)
        portfolio.on_fill(short_fill)
        cash_after_short = portfolio.cash

        # Cover (buy back)
        cover_fill = make_fill(Direction.LONG, 100, 440.0)
        portfolio.on_fill(cover_fill)

        # Cash decreases by (100 * 440) + 5 commission
        expected_cash = cash_after_short - (100 * 440.0) - 5.0
        assert portfolio.cash == pytest.approx(expected_cash)

    def test_cover_short_realizes_pnl(self, portfolio: Portfolio):
        """Covering short realizes P&L."""
        # Short at 450
        short_fill = make_fill(Direction.SHORT, 100, 450.0)
        portfolio.on_fill(short_fill)

        # Cover at 440 (profit for short)
        cover_fill = make_fill(Direction.LONG, 100, 440.0)
        portfolio.on_fill(cover_fill)

        pos = portfolio.get_position("SPY")
        # Realized P&L = (450 - 440) * 100 = 1000
        assert pos.realized_pnl == pytest.approx(1000.0)


class TestPortfolioEquity:
    """Test equity calculations."""

    def test_equity_equals_cash_plus_positions(self, portfolio: Portfolio):
        """Equity should equal cash + positions value."""
        # Buy some shares
        fill = make_fill(Direction.LONG, 100, 450.0)
        portfolio.on_fill(fill)

        # Update price
        portfolio.update_price("SPY", 460.0)

        # Equity = cash + position value
        expected_equity = portfolio.cash + (100 * 460.0)
        assert portfolio.equity() == pytest.approx(expected_equity)

    def test_equity_with_short_position(self, portfolio: Portfolio):
        """Equity calculation with short position."""
        # Short sell
        fill = make_fill(Direction.SHORT, 100, 450.0)
        portfolio.on_fill(fill)

        # Update price (price went down - good for short)
        portfolio.update_price("SPY", 440.0)

        # Position value is negative for short
        expected_equity = portfolio.cash + (-100 * 440.0)
        assert portfolio.equity() == pytest.approx(expected_equity)


class TestPortfolioSignalToOrder:
    """Test signal to order conversion."""

    def test_long_signal_creates_buy_order(
        self, portfolio: Portfolio, sample_bar: MarketDataEvent
    ):
        """LONG signal should create buy order."""
        signal = SignalEvent(
            timestamp=datetime(2024, 1, 15),
            symbol="SPY",
            direction=Direction.LONG,
            strength=1.0,
        )

        order = portfolio.signal_to_order(signal, sample_bar.close)

        assert order is not None
        assert order.direction == Direction.LONG
        assert order.symbol == "SPY"

    def test_position_sizing(self, portfolio: Portfolio, sample_bar: MarketDataEvent):
        """Order quantity should be based on position size percentage."""
        signal = SignalEvent(
            timestamp=datetime(2024, 1, 15),
            symbol="SPY",
            direction=Direction.LONG,
            strength=1.0,
        )

        order = portfolio.signal_to_order(signal, sample_bar.close)

        # 10% of 100k = 10k, at 450/share = 22 shares
        expected_qty = int(100_000 * 0.10 / sample_bar.close)
        assert order.quantity == expected_qty

    def test_signal_strength_affects_size(
        self, portfolio: Portfolio, sample_bar: MarketDataEvent
    ):
        """Signal strength should scale position size."""
        signal_half = SignalEvent(
            timestamp=datetime(2024, 1, 15),
            symbol="SPY",
            direction=Direction.LONG,
            strength=0.5,
        )

        signal_full = SignalEvent(
            timestamp=datetime(2024, 1, 15),
            symbol="SPY",
            direction=Direction.LONG,
            strength=1.0,
        )

        order_half = portfolio.signal_to_order(signal_half, sample_bar.close)

        portfolio2 = Portfolio(initial_capital=100_000, position_size_pct=0.10)
        order_full = portfolio2.signal_to_order(signal_full, sample_bar.close)

        # Half strength should give roughly half the quantity
        assert order_half.quantity < order_full.quantity


class TestPortfolioExposureAndLeverage:
    """Test gross_exposure, net_exposure, and leverage."""

    def test_gross_exposure_no_positions(self, portfolio: Portfolio):
        """Empty portfolio has zero gross exposure."""
        assert portfolio.gross_exposure() == 0.0

    def test_gross_exposure_long_only(self, portfolio: Portfolio):
        """Gross exposure sums absolute market values."""
        fill = make_fill(Direction.LONG, 100, 450.0)
        portfolio.on_fill(fill)
        portfolio.update_price("SPY", 460.0)

        assert portfolio.gross_exposure() == pytest.approx(100 * 460.0)

    def test_gross_exposure_long_and_short(self, portfolio: Portfolio):
        """Gross exposure is sum of absolutes, not net."""
        buy_fill = make_fill(Direction.LONG, 100, 450.0, symbol="SPY")
        portfolio.on_fill(buy_fill)
        portfolio.update_price("SPY", 450.0)

        short_fill = make_fill(Direction.SHORT, 50, 300.0, symbol="AAPL")
        portfolio.on_fill(short_fill)
        portfolio.update_price("AAPL", 300.0)

        # gross = |100*450| + |-50*300| = 45000 + 15000 = 60000
        assert portfolio.gross_exposure() == pytest.approx(60_000.0)

    def test_net_exposure_long_and_short(self, portfolio: Portfolio):
        """Net exposure cancels longs against shorts."""
        buy_fill = make_fill(Direction.LONG, 100, 450.0, symbol="SPY")
        portfolio.on_fill(buy_fill)
        portfolio.update_price("SPY", 450.0)

        short_fill = make_fill(Direction.SHORT, 50, 300.0, symbol="AAPL")
        portfolio.on_fill(short_fill)
        portfolio.update_price("AAPL", 300.0)

        # net = 100*450 + (-50*300) = 45000 - 15000 = 30000
        assert portfolio.net_exposure() == pytest.approx(30_000.0)

    def test_leverage_no_positions(self, portfolio: Portfolio):
        """Empty portfolio has zero leverage."""
        assert portfolio.leverage() == pytest.approx(0.0)

    def test_leverage_with_position(self, portfolio: Portfolio):
        """Leverage is gross_exposure / equity."""
        fill = make_fill(Direction.LONG, 100, 450.0)
        portfolio.on_fill(fill)
        portfolio.update_price("SPY", 450.0)

        # gross_exposure = 45000, equity = cash + positions
        # cash = 100000 - 45000 - 5 = 54995, positions = 45000, equity = 99995
        expected = 45_000.0 / portfolio.equity()
        assert portfolio.leverage() == pytest.approx(expected)


class TestPortfolioReset:
    """Test portfolio reset."""

    def test_reset_restores_initial_state(self, portfolio: Portfolio):
        """Reset should restore initial capital and clear positions."""
        # Make some trades
        fill = make_fill(Direction.LONG, 100, 450.0)
        portfolio.on_fill(fill)
        portfolio.record_equity(datetime(2024, 1, 15))

        # Reset
        portfolio.reset()

        assert portfolio.cash == 100_000
        assert portfolio.equity() == 100_000
        assert len(portfolio.positions) == 0
        assert len(portfolio.trades) == 0
        assert len(portfolio.equity_curve) == 0


class TestPortfolioEdgeCases:
    """Test edge cases for coverage."""

    def test_unrealized_pnl_flat_position(self):
        """Flat position returns 0 unrealized P&L."""
        pos = Position(symbol="SPY", quantity=0, avg_cost=0.0)
        assert pos.unrealized_pnl(450.0) == 0.0

    def test_repr(self, portfolio: Portfolio):
        """repr includes equity and cash info."""
        r = repr(portfolio)
        assert "Portfolio" in r
        assert "equity" in r
        assert "cash" in r

    def test_leverage_zero_equity(self):
        """Leverage returns inf when equity <= 0."""
        p = Portfolio(initial_capital=0.0)
        assert p.leverage() == float("inf")

    def test_signal_to_order_zero_price(self, portfolio: Portfolio):
        """Zero price returns None."""
        signal = SignalEvent(
            timestamp=datetime(2024, 1, 15),
            symbol="SPY",
            direction=Direction.LONG,
            strength=1.0,
        )
        order = portfolio.signal_to_order(signal, 0.0)
        assert order is None

    def test_signal_to_order_tiny_strength_yields_zero_qty(self):
        """Very small strength * equity / price rounds to 0 -> None."""
        p = Portfolio(initial_capital=100, position_size_pct=0.01)
        signal = SignalEvent(
            timestamp=datetime(2024, 1, 15),
            symbol="SPY",
            direction=Direction.LONG,
            strength=0.01,
        )
        order = p.signal_to_order(signal, 500.0)
        assert order is None

    def test_signal_to_order_already_long_enough(self, portfolio: Portfolio):
        """If already have enough long exposure, returns None."""
        fill = make_fill(Direction.LONG, 100, 450.0)
        portfolio.on_fill(fill)
        signal = SignalEvent(
            timestamp=datetime(2024, 1, 15),
            symbol="SPY",
            direction=Direction.LONG,
            strength=0.1,
        )
        order = portfolio.signal_to_order(signal, 450.0)
        assert order is None

    def test_signal_to_order_already_short_enough(self, portfolio: Portfolio):
        """If already have enough short exposure, returns None."""
        fill = make_fill(Direction.SHORT, 100, 450.0)
        portfolio.on_fill(fill)
        signal = SignalEvent(
            timestamp=datetime(2024, 1, 15),
            symbol="SPY",
            direction=Direction.SHORT,
            strength=0.1,
        )
        order = portfolio.signal_to_order(signal, 450.0)
        assert order is None

    def test_total_realized_pnl(self, portfolio: Portfolio):
        """total_realized_pnl sums across all positions."""
        buy = make_fill(Direction.LONG, 100, 450.0)
        portfolio.on_fill(buy)
        sell = make_fill(Direction.SHORT, 100, 460.0)
        portfolio.on_fill(sell)
        assert portfolio.total_realized_pnl() == pytest.approx(1000.0)

    def test_total_unrealized_pnl(self, portfolio: Portfolio):
        """total_unrealized_pnl computes across open positions."""
        fill = make_fill(Direction.LONG, 100, 450.0)
        portfolio.on_fill(fill)
        portfolio.update_price("SPY", 460.0)
        assert portfolio.total_unrealized_pnl() == pytest.approx(1000.0)

    def test_multiplier_with_universe(self):
        """Portfolio with universe uses contract_multiplier."""
        from sandtable.data.instrument import Future
        from sandtable.data.universe import Universe

        es = Future("ES", multiplier=50.0, tick_size=0.25)
        universe = Universe(instruments={"ES": es})
        p = Portfolio(initial_capital=100_000, universe=universe)
        assert p._multiplier("ES") == 50.0
        # unknown symbol falls back to 1.0
        assert p._multiplier("UNKNOWN") == 1.0

    def test_buy_covering_short_flips_long(self, portfolio: Portfolio):
        """Buying more than short position flips to long."""
        short = make_fill(Direction.SHORT, 50, 450.0)
        portfolio.on_fill(short)
        buy = make_fill(Direction.LONG, 100, 440.0)
        portfolio.on_fill(buy)
        pos = portfolio.get_position("SPY")
        assert pos.quantity == 50
        assert pos.is_long

    def test_sell_closing_long_flips_short(self, portfolio: Portfolio):
        """Selling more than long position flips to short."""
        buy = make_fill(Direction.LONG, 50, 450.0)
        portfolio.on_fill(buy)
        sell = make_fill(Direction.SHORT, 100, 460.0)
        portfolio.on_fill(sell)
        pos = portfolio.get_position("SPY")
        assert pos.quantity == -50
        assert pos.is_short
