"""tests/metrics/test_metrics.py

Tests for performance metrics calculation.
"""

from datetime import datetime, timedelta

import pytest

from backtester.core.events import Direction, FillEvent
from backtester.metrics.performance import (
    PerformanceMetrics,
    _calculate_cagr,
    _calculate_max_drawdown,
    _calculate_sharpe_ratio,
    _calculate_sortino_ratio,
    calculate_metrics,
)
from backtester.portfolio.portfolio import EquityPoint


def make_equity_curve(equities: list[float]) -> list[EquityPoint]:
    """
    Create equity curve from list of equity values.
    """
    base = datetime(2024, 1, 1)
    return [
        EquityPoint(
            timestamp=base + timedelta(days=i),
            equity=eq,
            cash=eq * 0.5,
            positions_value=eq * 0.5,
        )
        for i, eq in enumerate(equities)
    ]


def make_fill(
    direction: Direction,
    quantity: int,
    fill_price: float,
    day_offset: int = 0,
) -> FillEvent:
    """
    Create a fill event.
    """
    return FillEvent(
        timestamp=datetime(2024, 1, 1) + timedelta(days=day_offset),
        symbol="SPY",
        direction=direction,
        quantity=quantity,
        fill_price=fill_price,
        commission=5.0,
        slippage=0.5,
        market_impact=0.1,
    )


class TestCAGR:
    """
    Test CAGR calculation.
    """

    def test_positive_return(self):
        """
        CAGR for positive return over one year.
        """
        # 100k to 121k over 252 days = 21% annual return
        cagr = _calculate_cagr(100_000, 121_000, 252)
        assert cagr == pytest.approx(0.21, rel=0.01)

    def test_negative_return(self):
        """
        CAGR for negative return.
        """
        # 100k to 90k over 252 days
        cagr = _calculate_cagr(100_000, 90_000, 252)
        assert cagr == pytest.approx(-0.10, rel=0.01)

    def test_half_year(self):
        """
        CAGR annualizes correctly for partial year.
        """
        # 10% gain in half year should annualize to ~21%
        cagr = _calculate_cagr(100_000, 110_000, 126)
        assert cagr == pytest.approx(0.21, rel=0.02)

    def test_zero_start_equity(self):
        """
        CAGR handles zero start equity.
        """
        cagr = _calculate_cagr(0, 100_000, 252)
        assert cagr == 0.0


class TestMaxDrawdown:
    """
    Test max drawdown calculation.
    """

    def test_simple_drawdown(self):
        """
        Calculate drawdown from simple equity curve.
        """
        # Peak at 120, trough at 80 = 33.3% drawdown
        equities = [100, 110, 120, 115, 100, 90, 80, 90, 100]
        max_dd = _calculate_max_drawdown(equities)
        assert max_dd == pytest.approx(0.333, rel=0.01)

    def test_no_drawdown(self):
        """
        No drawdown for monotonically increasing equity.
        """
        equities = [100, 110, 120, 130, 140, 150]
        max_dd = _calculate_max_drawdown(equities)
        assert max_dd == 0.0

    def test_multiple_drawdowns(self):
        """
        Should return the maximum of multiple drawdowns.
        """
        # First drawdown: 100 to 90 = 10%
        # Second drawdown: 120 to 90 = 25%
        equities = [100, 90, 100, 110, 120, 90, 100]
        max_dd = _calculate_max_drawdown(equities)
        assert max_dd == pytest.approx(0.25, rel=0.01)

    def test_single_point(self):
        """
        Handle single point equity curve.
        """
        max_dd = _calculate_max_drawdown([100])
        assert max_dd == 0.0


class TestSharpeRatio:
    """
    Test Sharpe ratio calculation.
    """

    def test_positive_sharpe(self):
        """
        Positive Sharpe for positive risk-adjusted returns.
        """
        # Consistent small positive returns
        returns = [0.001] * 100 + [0.002] * 100  # Mix of returns
        sharpe = _calculate_sharpe_ratio(returns, 0.0)
        assert sharpe > 0

    def test_negative_sharpe(self):
        """
        Negative Sharpe for negative returns.
        """
        # Consistent small negative returns
        returns = [-0.001] * 100 + [-0.002] * 100
        sharpe = _calculate_sharpe_ratio(returns, 0.0)
        assert sharpe < 0

    def test_insufficient_data(self):
        """
        Handle insufficient data.
        """
        sharpe = _calculate_sharpe_ratio([0.01], 0.0)
        assert sharpe == 0.0


class TestSortinoRatio:
    """
    Test Sortino ratio calculation.
    """

    def test_sortino_higher_than_sharpe_for_positive_skew(self):
        """
        Sortino should be higher when downside volatility is low.
        """
        # Returns with some large positive, small negative
        returns = [0.02, 0.03, 0.01, -0.005, 0.02, -0.003, 0.015]
        sharpe = _calculate_sharpe_ratio(returns, 0.0)
        sortino = _calculate_sortino_ratio(returns, 0.0)

        # Sortino should be >= Sharpe when positive skew
        assert sortino >= sharpe

    def test_no_negative_returns(self):
        """
        Handle case with no negative returns.
        """
        returns = [0.01, 0.02, 0.015, 0.01]
        sortino = _calculate_sortino_ratio(returns, 0.0)
        # Should return high value (capped at 10)
        assert sortino == 10.0


class TestTradeMetrics:
    """
    Test trade-based metrics.
    """

    def test_win_rate_calculation(self):
        """
        Win rate should be correct.
        """
        trades = [
            # Winning trade: buy at 100, sell at 110
            make_fill(Direction.LONG, 100, 100.0, 0),
            make_fill(Direction.SHORT, 100, 110.0, 1),
            # Losing trade: buy at 100, sell at 95
            make_fill(Direction.LONG, 100, 100.0, 2),
            make_fill(Direction.SHORT, 100, 95.0, 3),
        ]

        equity_curve = make_equity_curve([100_000] * 10)
        metrics = calculate_metrics(equity_curve, trades)

        # 1 win, 1 loss = 50% win rate
        assert metrics.win_rate == pytest.approx(0.5)

    def test_profit_factor(self):
        """
        Profit factor should be gross profit / gross loss.
        """
        trades = [
            # Win $1000: buy 100 @ 100, sell @ 110
            make_fill(Direction.LONG, 100, 100.0, 0),
            make_fill(Direction.SHORT, 100, 110.0, 1),
            # Lose $500: buy 100 @ 100, sell @ 95
            make_fill(Direction.LONG, 100, 100.0, 2),
            make_fill(Direction.SHORT, 100, 95.0, 3),
        ]

        equity_curve = make_equity_curve([100_000] * 10)
        metrics = calculate_metrics(equity_curve, trades)

        # Profit factor = (110-100)*100 / (100-95)*100 = 1000/500 = 2.0
        # (minus commissions, so slightly different)
        assert metrics.profit_factor > 1.5

    def test_num_trades(self):
        """
        Should count round-trip trades correctly.
        """
        trades = [
            make_fill(Direction.LONG, 100, 100.0, 0),
            make_fill(Direction.SHORT, 100, 110.0, 1),
            make_fill(Direction.LONG, 50, 105.0, 2),
            make_fill(Direction.SHORT, 50, 108.0, 3),
        ]

        equity_curve = make_equity_curve([100_000] * 10)
        metrics = calculate_metrics(equity_curve, trades)

        assert metrics.num_trades == 2

    def test_no_trades(self):
        """
        Handle case with no trades.
        """
        equity_curve = make_equity_curve([100_000, 100_100, 100_200])
        metrics = calculate_metrics(equity_curve, [])

        assert metrics.num_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.profit_factor == 0.0


class TestCalculateMetrics:
    """
    Test full metrics calculation.
    """

    def test_returns_performance_metrics(self):
        """
        Should return PerformanceMetrics instance.
        """
        equity_curve = make_equity_curve([100_000, 100_500, 101_000])
        metrics = calculate_metrics(equity_curve, [])

        assert isinstance(metrics, PerformanceMetrics)

    def test_total_return(self):
        """
        Total return should be (end - start) / start.
        """
        equity_curve = make_equity_curve([100_000, 105_000, 110_000])
        metrics = calculate_metrics(equity_curve, [])

        expected_return = (110_000 - 100_000) / 100_000
        assert metrics.total_return == pytest.approx(expected_return)

    def test_start_end_equity(self):
        """
        Should capture start and end equity.
        """
        equity_curve = make_equity_curve([100_000, 105_000, 110_000])
        metrics = calculate_metrics(equity_curve, [])

        assert metrics.start_equity == 100_000
        assert metrics.end_equity == 110_000

    def test_num_days(self):
        """
        Should count number of days.
        """
        equity_curve = make_equity_curve([100_000] * 252)
        metrics = calculate_metrics(equity_curve, [])

        assert metrics.num_days == 252

    def test_empty_equity_curve(self):
        """
        Handle empty equity curve.
        """
        metrics = calculate_metrics([], [])

        assert metrics.total_return == 0.0
        assert metrics.num_trades == 0

    def test_single_point_equity_curve(self):
        """
        Handle single point equity curve.
        """
        equity_curve = make_equity_curve([100_000])
        metrics = calculate_metrics(equity_curve, [])

        assert metrics.total_return == 0.0


class TestPerformanceMetricsStr:
    """
    Test PerformanceMetrics string representation.
    """

    def test_str_format(self):
        """
        String representation should be readable.
        """
        metrics = PerformanceMetrics(
            total_return=0.10,
            cagr=0.12,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=0.15,
            num_trades=50,
            win_rate=0.55,
            profit_factor=1.8,
            avg_trade_pnl=100.0,
            start_equity=100_000,
            end_equity=110_000,
            num_days=252,
        )

        str_repr = str(metrics)

        assert "total_return" in str_repr
        assert "10.00%" in str_repr
        assert "sharpe_ratio" in str_repr
        assert "1.50" in str_repr
        assert "win_rate" in str_repr
        assert "55.0%" in str_repr
