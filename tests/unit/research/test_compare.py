"""
tests/unit/research/test_compare.py

Tests for strategy comparison.
"""

import pytest

from sandtable.api import run_backtest
from sandtable.core.events import Direction, MarketDataEvent, SignalEvent
from sandtable.research.compare import ComparisonResult, run_comparison
from sandtable.strategy.abstract_strategy import AbstractStrategy
from tests.conftest import make_data_handler


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


def _run_strategy(fast: int, slow: int):
    """Helper to run a strategy and return the result."""
    data = make_data_handler(["SPY"])
    return run_backtest(
        strategy=SimpleMAStrategy(fast_period=fast, slow_period=slow),
        data=data,
    )


class TestRunComparison:
    def test_basic_comparison(self):
        """Two strategies produce a valid ComparisonResult."""
        results = {
            "MA(5,20)": _run_strategy(5, 20),
            "MA(10,30)": _run_strategy(10, 30),
        }
        comp = run_comparison(results)

        assert isinstance(comp, ComparisonResult)
        assert len(comp.results) == 2

    def test_performance_table_shape(self):
        """Performance table has one row per strategy."""
        results = {
            "A": _run_strategy(5, 20),
            "B": _run_strategy(10, 30),
        }
        comp = run_comparison(results)

        assert len(comp.performance_table) == 2
        assert "A" in comp.performance_table.index
        assert "B" in comp.performance_table.index
        assert "sharpe_ratio" in comp.performance_table.columns

    def test_correlation_matrix_symmetric(self):
        """Correlation matrix should be symmetric with 1.0 on diagonal."""
        results = {
            "A": _run_strategy(5, 20),
            "B": _run_strategy(10, 30),
        }
        comp = run_comparison(results)

        corr = comp.correlation_matrix
        assert corr.shape == (2, 2)
        # diagonal should be 1.0
        assert abs(corr.loc["A", "A"] - 1.0) < 1e-10
        assert abs(corr.loc["B", "B"] - 1.0) < 1e-10
        # symmetric
        assert abs(corr.loc["A", "B"] - corr.loc["B", "A"]) < 1e-10

    def test_identical_strategies_correlation_one(self):
        """Two identical strategies should have correlation ~1.0."""
        result_a = _run_strategy(10, 30)
        # run_backtest is deterministic, so same params = same result
        result_b = _run_strategy(10, 30)

        comp = run_comparison({"A": result_a, "B": result_b})
        corr_ab = comp.correlation_matrix.loc["A", "B"]
        assert abs(corr_ab - 1.0) < 1e-6

    def test_blended_equity_curve(self):
        """Blended curve should have entries."""
        results = {
            "A": _run_strategy(5, 20),
            "B": _run_strategy(10, 30),
        }
        comp = run_comparison(results)

        assert len(comp.blended_equity_curve) > 0

    def test_single_strategy(self):
        """Comparison with a single strategy should work."""
        results = {"Solo": _run_strategy(10, 30)}
        comp = run_comparison(results)

        assert len(comp.performance_table) == 1
        assert comp.correlation_matrix.shape == (1, 1)

    def test_empty_raises(self):
        """Empty results dict should raise."""
        with pytest.raises(ValueError, match="at least one"):
            run_comparison({})
