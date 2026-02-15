"""
tests/core/test_result.py

Tests for BacktestResult.
"""

from datetime import datetime

import pandas as pd
import pytest

from sandtable.core.events import Direction, FillEvent
from sandtable.core.result import BacktestResult
from sandtable.metrics.performance import PerformanceMetrics
from sandtable.portfolio.portfolio import EquityPoint


@pytest.fixture
def sample_result():
    metrics = PerformanceMetrics(
        total_return=0.05,
        cagr=0.10,
        sharpe_ratio=1.5,
        sortino_ratio=2.0,
        max_drawdown=0.08,
        num_fills=6,
        num_trades=3,
        win_rate=0.67,
        profit_factor=2.0,
        avg_trade_pnl=500.0,
        start_equity=100_000.0,
        end_equity=105_000.0,
        num_days=252,
    )
    equity_curve = [
        EquityPoint(timestamp=datetime(2023, 1, 3), equity=100_000, cash=100_000, positions_value=0),
        EquityPoint(timestamp=datetime(2023, 1, 4), equity=101_000, cash=90_000, positions_value=11_000),
        EquityPoint(timestamp=datetime(2023, 1, 5), equity=105_000, cash=105_000, positions_value=0),
    ]
    trades = [
        FillEvent(
            timestamp=datetime(2023, 1, 3),
            symbol="SPY",
            direction=Direction.LONG,
            quantity=100,
            fill_price=400.0,
            commission=1.0,
            slippage=0.02,
            market_impact=0.0,
        ),
        FillEvent(
            timestamp=datetime(2023, 1, 5),
            symbol="SPY",
            direction=Direction.SHORT,
            quantity=100,
            fill_price=450.0,
            commission=1.0,
            slippage=0.02,
            market_impact=0.0,
        ),
    ]
    price_df = pd.DataFrame({
        "timestamp": [datetime(2023, 1, 3), datetime(2023, 1, 4), datetime(2023, 1, 5)],
        "open": [399, 401, 449],
        "high": [405, 405, 455],
        "low": [395, 398, 445],
        "close": [400, 402, 450],
        "volume": [1e6, 1.1e6, 1.2e6],
    })

    return BacktestResult(
        metrics=metrics,
        equity_curve=equity_curve,
        trades=trades,
        parameters={"strategy": "test", "initial_capital": 100_000},
        price_data={"SPY": price_df},
        symbols=["SPY"],
        initial_capital=100_000.0,
        start_date=datetime(2023, 1, 3),
        end_date=datetime(2023, 1, 5),
    )


class TestBacktestResult:
    def test_is_frozen(self, sample_result):
        with pytest.raises(AttributeError):
            sample_result.initial_capital = 200_000

    def test_equity_dataframe(self, sample_result):
        df = sample_result.equity_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == ["timestamp", "equity", "cash", "positions_value"]
        assert df["equity"].iloc[-1] == 105_000

    def test_trades_dataframe(self, sample_result):
        df = sample_result.trades_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "direction" in df.columns
        assert df["fill_price"].iloc[0] == 400.0

    def test_trades_dataframe_empty(self):
        result = BacktestResult(
            metrics=PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            equity_curve=[],
            trades=[],
            parameters={},
            price_data={},
            symbols=[],
            initial_capital=100_000,
            start_date=None,
            end_date=None,
        )
        df = result.trades_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_str(self, sample_result):
        s = str(sample_result)
        assert "SPY" in s
        assert "BacktestResult" in s

    def test_attributes(self, sample_result):
        assert sample_result.symbols == ["SPY"]
        assert sample_result.initial_capital == 100_000.0
        assert sample_result.metrics.sharpe_ratio == 1.5
