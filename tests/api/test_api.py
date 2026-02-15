"""
tests/api/test_api.py

Tests for the research API (run_backtest, run_parameter_sweep).
"""

from dataclasses import dataclass
from pathlib import Path

import pytest

from sandtable.api import run_backtest, run_parameter_sweep, SweepResult
from sandtable.metrics import Metric
from sandtable.core.events import Direction, MarketDataEvent, SignalEvent
from sandtable.core.result import BacktestResult
from sandtable.data_handlers.csv_data_handler import CSVDataHandler
from sandtable.data_handlers.multi_handler import MultiDataHandler
from sandtable.execution.simulator import ExecutionConfig
from sandtable.execution.slippage import FixedSlippage
from sandtable.strategy.abstract_strategy import AbstractStrategy

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "sample_ohlcv.csv"


@dataclass
class SimpleMAStrategy(AbstractStrategy):
    """Simple moving average crossover for testing."""

    fast_period: int = 10
    slow_period: int = 30

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


class TestRunBacktest:
    def test_basic_csv(self):
        data = CSVDataHandler(DATA_PATH, "SPY")
        result = run_backtest(
            strategy=SimpleMAStrategy(),
            data=data,
        )
        assert isinstance(result, BacktestResult)
        assert result.symbols == ["SPY"]
        assert result.initial_capital == 100_000.0
        assert result.metrics.num_days > 0
        assert len(result.equity_curve) > 0

    def test_with_slippage(self):
        data = CSVDataHandler(DATA_PATH, "SPY")
        result = run_backtest(
            strategy=SimpleMAStrategy(),
            data=data,
            slippage=FixedSlippage(bps=10),
        )
        assert isinstance(result, BacktestResult)

    def test_with_commission_float(self):
        data = CSVDataHandler(DATA_PATH, "SPY")
        result = run_backtest(
            strategy=SimpleMAStrategy(),
            data=data,
            commission=0.01,
        )
        assert isinstance(result, BacktestResult)

    def test_with_execution_config(self):
        data = CSVDataHandler(DATA_PATH, "SPY")
        result = run_backtest(
            strategy=SimpleMAStrategy(),
            data=data,
            commission=ExecutionConfig(commission_per_share=0.01, commission_minimum=2.0),
        )
        assert isinstance(result, BacktestResult)

    def test_custom_capital(self):
        data = CSVDataHandler(DATA_PATH, "SPY")
        result = run_backtest(
            strategy=SimpleMAStrategy(),
            data=data,
            initial_capital=50_000.0,
        )
        assert result.initial_capital == 50_000.0

    def test_multi_symbol_csv(self):
        data = MultiDataHandler({
            "SPY": CSVDataHandler(DATA_PATH, "SPY"),
            "SPY2": CSVDataHandler(DATA_PATH, "SPY2"),
        })
        result = run_backtest(
            strategy=SimpleMAStrategy(),
            data=data,
        )
        assert isinstance(result, BacktestResult)
        assert set(result.symbols) == {"SPY", "SPY2"}
        assert len(result.price_data) == 2

    def test_equity_dataframe(self):
        data = CSVDataHandler(DATA_PATH, "SPY")
        result = run_backtest(
            strategy=SimpleMAStrategy(),
            data=data,
        )
        df = result.equity_dataframe()
        assert "equity" in df.columns
        assert len(df) > 0

    def test_trades_dataframe(self):
        data = CSVDataHandler(DATA_PATH, "SPY")
        result = run_backtest(
            strategy=SimpleMAStrategy(),
            data=data,
        )
        df = result.trades_dataframe()
        assert "fill_price" in df.columns


class TestParameterSweep:
    def test_basic_sweep(self):
        data = CSVDataHandler(DATA_PATH, "SPY")
        result = run_parameter_sweep(
            strategy_class=SimpleMAStrategy,
            param_grid={"fast_period": [5, 10], "slow_period": [20, 30]},
            data=data,
        )
        assert isinstance(result, SweepResult)
        assert len(result.results) == 4
        assert len(result.param_combinations) == 4

    def test_best_params(self):
        data = CSVDataHandler(DATA_PATH, "SPY")
        result = run_parameter_sweep(
            strategy_class=SimpleMAStrategy,
            param_grid={"fast_period": [5, 10], "slow_period": [20, 30]},
            data=data,
        )
        best = result.best_params
        assert "fast_period" in best
        assert "slow_period" in best

    def test_to_dataframe(self):
        data = CSVDataHandler(DATA_PATH, "SPY")
        result = run_parameter_sweep(
            strategy_class=SimpleMAStrategy,
            param_grid={"fast_period": [5, 10], "slow_period": [20, 30]},
            data=data,
        )
        df = result.to_dataframe()
        assert len(df) == 4
        assert "fast_period" in df.columns
        assert Metric.SHARPE_RATIO in df.columns

    def test_heatmap_data(self):
        data = CSVDataHandler(DATA_PATH, "SPY")
        result = run_parameter_sweep(
            strategy_class=SimpleMAStrategy,
            param_grid={"fast_period": [5, 10], "slow_period": [20, 30]},
            data=data,
        )
        heatmap = result.heatmap_data("fast_period", "slow_period")
        assert heatmap.shape == (2, 2)
