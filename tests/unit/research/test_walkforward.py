"""
tests/unit/research/test_walkforward.py

Tests for walk-forward analysis.
"""

import pytest

from sandtable.core.events import Direction, MarketDataEvent, SignalEvent
from sandtable.research.walkforward import WalkForwardFold, WalkForwardResult, run_walkforward
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


@pytest.fixture
def full_data():
    """Data handler with all test data loaded."""
    return make_data_handler(["SPY"])


class TestWalkForward:
    def test_basic_walkforward(self, full_data):
        """Walk-forward produces folds with correct structure."""
        result = run_walkforward(
            strategy_cls=SimpleMAStrategy,
            param_grid={"fast_period": [5, 10], "slow_period": [20, 30]},
            data=full_data,
            train_window=100,
            test_window=50,
        )
        assert isinstance(result, WalkForwardResult)
        assert len(result.folds) >= 1
        for fold in result.folds:
            assert isinstance(fold, WalkForwardFold)
            assert fold.train_start <= fold.train_end
            assert fold.test_start <= fold.test_end
            assert fold.train_end < fold.test_start

    def test_step_size_defaults_to_test_window(self, full_data):
        """When step_size is None, folds don't overlap in test periods."""
        result = run_walkforward(
            strategy_cls=SimpleMAStrategy,
            param_grid={"fast_period": [5, 10], "slow_period": [20]},
            data=full_data,
            train_window=100,
            test_window=50,
        )
        if len(result.folds) >= 2:
            for i in range(1, len(result.folds)):
                # test periods should not overlap
                assert result.folds[i].test_start >= result.folds[i - 1].test_end

    def test_chosen_params_are_from_grid(self, full_data):
        """Chosen parameters must be from the param grid."""
        result = run_walkforward(
            strategy_cls=SimpleMAStrategy,
            param_grid={"fast_period": [5, 10], "slow_period": [20, 30]},
            data=full_data,
            train_window=100,
            test_window=50,
        )
        for fold in result.folds:
            assert fold.chosen_params["fast_period"] in [5, 10]
            assert fold.chosen_params["slow_period"] in [20, 30]

    def test_oos_equity_curve_populated(self, full_data):
        """Stitched OOS equity curve should have data."""
        result = run_walkforward(
            strategy_cls=SimpleMAStrategy,
            param_grid={"fast_period": [5, 10], "slow_period": [20]},
            data=full_data,
            train_window=100,
            test_window=50,
        )
        assert len(result.oos_equity_curve) > 0

    def test_aggregate_metrics_populated(self, full_data):
        """Aggregate OOS metrics should be computed."""
        result = run_walkforward(
            strategy_cls=SimpleMAStrategy,
            param_grid={"fast_period": [5, 10], "slow_period": [20]},
            data=full_data,
            train_window=100,
            test_window=50,
        )
        # metrics should be numbers (may be 0 if no trades)
        assert isinstance(result.oos_sharpe, float)
        assert isinstance(result.oos_max_drawdown, float)
        assert isinstance(result.oos_cagr, float)

    def test_config_recorded(self, full_data):
        """Config dict should capture the run settings."""
        result = run_walkforward(
            strategy_cls=SimpleMAStrategy,
            param_grid={"fast_period": [5, 10], "slow_period": [20]},
            data=full_data,
            train_window=100,
            test_window=50,
        )
        assert result.config["train_window"] == 100
        assert result.config["test_window"] == 50
        assert result.config["step_size"] == 50  # defaults to test_window

    def test_insufficient_data_raises(self, full_data):
        """Should raise ValueError if data too short for train + test."""
        with pytest.raises(ValueError, match="Insufficient data"):
            run_walkforward(
                strategy_cls=SimpleMAStrategy,
                param_grid={"fast_period": [5]},
                data=full_data,
                train_window=999_999,
                test_window=999_999,
            )

    def test_fold_metrics_present(self, full_data):
        """Each fold should have IS and OOS metrics dicts."""
        result = run_walkforward(
            strategy_cls=SimpleMAStrategy,
            param_grid={"fast_period": [5, 10], "slow_period": [20]},
            data=full_data,
            train_window=100,
            test_window=50,
        )
        for fold in result.folds:
            assert "sharpe_ratio" in fold.in_sample_metrics
            assert "sharpe_ratio" in fold.out_of_sample_metrics
