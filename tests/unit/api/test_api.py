"""
tests/api/test_api.py

Tests for the research API (run_backtest, run_parameter_sweep).
"""

from sandtable.api import SweepResult, run_backtest, run_parameter_sweep
from sandtable.core.events import Direction, MarketDataEvent, SignalEvent
from sandtable.core.result import BacktestResult
from sandtable.data_types.metric import Metric
from sandtable.execution.simulator import ExecutionConfig
from sandtable.execution.slippage import FixedSlippage
from sandtable.metrics.performance import PerformanceMetrics
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


class TestRunBacktest:
    def test_basic_csv(self):
        data = make_data_handler(["SPY"])
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
        data = make_data_handler(["SPY"])
        result = run_backtest(
            strategy=SimpleMAStrategy(),
            data=data,
            slippage=FixedSlippage(bps=10),
        )
        assert isinstance(result, BacktestResult)

    def test_with_commission_float(self):
        data = make_data_handler(["SPY"])
        result = run_backtest(
            strategy=SimpleMAStrategy(),
            data=data,
            commission=0.01,
        )
        assert isinstance(result, BacktestResult)

    def test_with_execution_config(self):
        data = make_data_handler(["SPY"])
        result = run_backtest(
            strategy=SimpleMAStrategy(),
            data=data,
            commission=ExecutionConfig(commission_per_share=0.01, commission_minimum=2.0),
        )
        assert isinstance(result, BacktestResult)

    def test_custom_capital(self):
        data = make_data_handler(["SPY"])
        result = run_backtest(
            strategy=SimpleMAStrategy(),
            data=data,
            initial_capital=50_000.0,
        )
        assert result.initial_capital == 50_000.0

    def test_multi_symbol_csv(self):
        data = make_data_handler(["SPY", "SPY2"])
        result = run_backtest(
            strategy=SimpleMAStrategy(),
            data=data,
        )
        assert isinstance(result, BacktestResult)
        assert set(result.symbols) == {"SPY", "SPY2"}
        assert len(result.price_data) == 2

    def test_equity_dataframe(self):
        data = make_data_handler(["SPY"])
        result = run_backtest(
            strategy=SimpleMAStrategy(),
            data=data,
        )
        df = result.equity_dataframe()
        assert "equity" in df.columns
        assert len(df) > 0

    def test_trades_dataframe(self):
        data = make_data_handler(["SPY"])
        result = run_backtest(
            strategy=SimpleMAStrategy(),
            data=data,
        )
        df = result.trades_dataframe()
        assert "fill_price" in df.columns


class TestParameterSweep:
    def test_basic_sweep(self):
        data = make_data_handler(["SPY"])
        result = run_parameter_sweep(
            strategy_class=SimpleMAStrategy,
            param_grid={"fast_period": [5, 10], "slow_period": [20, 30]},
            data=data,
        )
        assert isinstance(result, SweepResult)
        assert len(result.results) == 4
        assert len(result.param_combinations) == 4

    def test_best_params(self):
        data = make_data_handler(["SPY"])
        result = run_parameter_sweep(
            strategy_class=SimpleMAStrategy,
            param_grid={"fast_period": [5, 10], "slow_period": [20, 30]},
            data=data,
        )
        best = result.best_params
        assert "fast_period" in best
        assert "slow_period" in best

    def test_to_dataframe(self):
        data = make_data_handler(["SPY"])
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
        data = make_data_handler(["SPY"])
        result = run_parameter_sweep(
            strategy_class=SimpleMAStrategy,
            param_grid={"fast_period": [5, 10], "slow_period": [20, 30]},
            data=data,
        )
        heatmap = result.heatmap_data("fast_period", "slow_period")
        assert heatmap.shape == (2, 2)


class TestBestIndexDirection:
    """Verify _best_index picks max for most metrics, min for MAX_DRAWDOWN."""

    def _make_sweep_result(self, metric, drawdowns, sharpes):
        """Build a SweepResult from hand-crafted metric values."""
        results = []
        params = []
        for i, (dd, sr) in enumerate(zip(drawdowns, sharpes)):
            # create a minimal BacktestResult with controlled metrics
            mock_metrics = PerformanceMetrics(
                total_return=0.0,
                cagr=0.0,
                sharpe_ratio=sr,
                sortino_ratio=0.0,
                max_drawdown=dd,
                num_fills=0,
                num_trades=0,
                win_rate=0.0,
                profit_factor=0.0,
                avg_trade_pnl=0.0,
                start_equity=100_000,
                end_equity=100_000,
                num_days=252,
            )
            result = BacktestResult(
                symbols=["SPY"],
                initial_capital=100_000,
                equity_curve=[],
                trades=[],
                price_data={},
                metrics=mock_metrics,
                parameters={},
                start_date=None,
                end_date=None,
            )
            results.append(result)
            params.append({"idx": i})

        return SweepResult(results=results, param_combinations=params, metric=metric)

    def test_best_sharpe_picks_maximum(self):
        sweep = self._make_sweep_result(
            Metric.SHARPE_RATIO,
            drawdowns=[0.10, 0.20, 0.05],
            sharpes=[1.0, 2.5, 1.8],
        )
        assert sweep.best_params == {"idx": 1}  # sharpe=2.5

    def test_best_drawdown_picks_minimum(self):
        sweep = self._make_sweep_result(
            Metric.MAX_DRAWDOWN,
            drawdowns=[0.10, 0.20, 0.05],
            sharpes=[1.0, 2.5, 1.8],
        )
        assert sweep.best_params == {"idx": 2}  # dd=0.05 is best

    def test_best_drawdown_does_not_pick_worst(self):
        """Regression: old code always maximized, picking worst drawdown."""
        sweep = self._make_sweep_result(
            Metric.MAX_DRAWDOWN,
            drawdowns=[0.05, 0.30],
            sharpes=[0.0, 0.0],
        )
        # the best drawdown is 0.05, NOT 0.30
        best = sweep.best_result
        assert best.metrics.max_drawdown == 0.05
