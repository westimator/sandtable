"""
tests/unit/core/test_backtest.py

Tests for the Backtest engine edge cases.
"""

from sandtable.api import run_backtest
from sandtable.core.events import Direction, MarketDataEvent, SignalEvent
from sandtable.strategy.abstract_strategy import AbstractStrategy
from tests.conftest import make_data_handler


class _AlwaysLongStrategy(AbstractStrategy):
    """Goes long on every bar after warmup."""

    def __init__(self, *, lookback: int = 5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.lookback = lookback

    def generate_signal(self, bar: MarketDataEvent) -> SignalEvent | None:
        if self.symbol_bar_count(bar.symbol) < self.lookback:
            return None
        return SignalEvent(
            timestamp=bar.timestamp,
            symbol=bar.symbol,
            direction=Direction.LONG,
            strength=1.0,
        )


class TestBacktestReset:
    def test_reset_allows_rerun(self):
        """After reset(), backtest can be run again."""
        from sandtable.core.backtest import Backtest
        from sandtable.execution.simulator import ExecutionSimulator
        from sandtable.portfolio.portfolio import Portfolio

        data = make_data_handler(["SPY"])
        strategy = _AlwaysLongStrategy(lookback=5)
        portfolio = Portfolio(initial_capital=100_000)
        executor = ExecutionSimulator()
        bt = Backtest(data, strategy, portfolio, executor)

        m1 = bt.run()
        assert m1.num_days > 0

        bt.reset()
        m2 = bt.run()
        assert m2.num_days > 0

    def test_run_returns_metrics(self):
        data = make_data_handler(["SPY"])
        result = run_backtest(strategy=_AlwaysLongStrategy(lookback=5), data=data)
        assert result.metrics is not None
        assert result.metrics.num_days > 0
