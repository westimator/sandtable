"""
src/sandtable/api.py

High-level research API for running backtests with minimal boilerplate.

Provides `run_backtest()` one-liner and `run_parameter_sweep()` for
parameter optimization.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import pandas as pd

from sandtable.core.backtest import Backtest
from sandtable.core.result import BacktestResult
from sandtable.data_handlers.abstract_data_handler import AbstractDataHandler
from sandtable.execution.impact import MarketImpactModel
from sandtable.execution.simulator import ExecutionConfig, ExecutionSimulator
from sandtable.execution.slippage import SlippageModel
from sandtable.metrics import Metric
from sandtable.portfolio.portfolio import Portfolio
from sandtable.strategy.abstract_strategy import AbstractStrategy
from sandtable.utils.logger import get_logger

logger = get_logger(__name__)


def run_backtest(
    strategy: AbstractStrategy,
    data: AbstractDataHandler,
    initial_capital: float = 100_000.0,
    position_size_pct: float = 0.10,
    slippage: SlippageModel | None = None,
    impact: MarketImpactModel | None = None,
    commission: float | ExecutionConfig | None = None,
) -> BacktestResult:
    """
    Run a backtest with minimal boilerplate.

    Args:
        strategy: AbstractStrategy instance to backtest
        data: Data handler providing market data bars
        initial_capital: Starting capital
        position_size_pct: Fraction of equity per trade
        slippage: Optional slippage model
        impact: Optional market impact model
        commission: Per-share commission float or ExecutionConfig

    Returns:
        BacktestResult with all outputs
    """
    price_data = data.get_price_data()
    symbols_list = sorted(price_data.keys())

    logger.info("Running backtest: %s on %s", type(strategy).__name__, symbols_list)

    # Build execution simulator
    executor = _build_executor(slippage, impact, commission)

    # Build portfolio
    portfolio = Portfolio(initial_capital=initial_capital, position_size_pct=position_size_pct)

    # Run backtest
    backtest = Backtest(
        data_handler=data,
        strategy=strategy,
        portfolio=portfolio,
        executor=executor,
    )
    metrics = backtest.run()

    # Determine date range from equity curve
    eq = portfolio.equity_curve
    start_date = eq[0].timestamp if eq else None
    end_date = eq[-1].timestamp if eq else None

    # Build parameters dict
    parameters = {
        "strategy": type(strategy).__name__,
        "symbols": symbols_list,
        "initial_capital": initial_capital,
        "position_size_pct": position_size_pct,
    }

    logger.info("Backtest finished: return=%.2f%%, sharpe=%.2f", metrics.total_return * 100, metrics.sharpe_ratio)

    return BacktestResult(
        metrics=metrics,
        equity_curve=portfolio.equity_curve,
        trades=portfolio.trades,
        parameters=parameters,
        price_data=price_data,
        symbols=symbols_list,
        initial_capital=initial_capital,
        start_date=start_date,
        end_date=end_date,
    )


def _build_executor(
    slippage: SlippageModel | None,
    impact: MarketImpactModel | None,
    commission: float | ExecutionConfig | None,
) -> ExecutionSimulator:
    """Build ExecutionSimulator from convenience args."""
    kwargs: dict[str, Any] = {}

    if slippage is not None:
        kwargs["slippage_model"] = slippage
    if impact is not None:
        kwargs["impact_model"] = impact

    if isinstance(commission, ExecutionConfig):
        kwargs["config"] = commission
    elif isinstance(commission, (int, float)):
        kwargs["config"] = ExecutionConfig(commission_per_share=commission)

    return ExecutionSimulator(**kwargs)


@dataclass
class SweepResult:
    """Container for parameter sweep results."""

    results: list[BacktestResult]
    param_combinations: list[dict[str, Any]]
    metric: Metric = Metric.SHARPE_RATIO

    @property
    def best_params(self) -> dict[str, Any]:
        idx = self._best_index()
        return self.param_combinations[idx]

    @property
    def best_result(self) -> BacktestResult:
        return self.results[self._best_index()]

    def _best_index(self) -> int:
        values = [getattr(r.metrics, self.metric) for r in self.results]
        return max(range(len(values)), key=lambda i: values[i])

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for params, result in zip(self.param_combinations, self.results):
            row = dict(params)
            row[Metric.TOTAL_RETURN] = result.metrics.total_return
            row[Metric.SHARPE_RATIO] = result.metrics.sharpe_ratio
            row[Metric.SORTINO_RATIO] = result.metrics.sortino_ratio
            row[Metric.MAX_DRAWDOWN] = result.metrics.max_drawdown
            row["num_trades"] = result.metrics.num_trades
            row[Metric.WIN_RATE] = result.metrics.win_rate
            rows.append(row)
        return pd.DataFrame(rows)

    def heatmap_data(self, row_param: str, col_param: str) -> pd.DataFrame:
        df = self.to_dataframe()
        return df.pivot_table(
            index=row_param,
            columns=col_param,
            values=self.metric,
            aggfunc="first",
        )


def run_parameter_sweep(
    strategy_class: type[AbstractStrategy],
    param_grid: dict[str, list[Any]],
    data: AbstractDataHandler,
    metric: Metric = Metric.SHARPE_RATIO,
    **backtest_kwargs: Any,
) -> SweepResult:
    """
    Run a parameter sweep over a strategy class.

    Args:
        strategy_class: AbstractStrategy class (not instance) to instantiate per combo
        param_grid: Dict mapping parameter names to lists of values
        data: Data handler providing market data bars (reset between runs)
        metric: Metric to optimize (default Metric.SHARPE_RATIO)
        **backtest_kwargs: Additional kwargs for run_backtest()

    Returns:
        SweepResult with all results and best parameters
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    total_combos = 1
    for v in param_values:
        total_combos *= len(v)
    logger.info("Starting parameter sweep: %d combinations over %s", total_combos, param_names)

    results: list[BacktestResult] = []
    combinations: list[dict[str, Any]] = []

    for combo in itertools.product(*param_values):
        params = dict(zip(param_names, combo))
        combinations.append(params)

        data.reset()
        strategy = strategy_class(**params)
        result = run_backtest(
            strategy=strategy,
            data=data,
            **backtest_kwargs,
        )
        results.append(result)

        logger.debug("Sweep: %s -> %s=%.4f", params, metric, getattr(result.metrics, metric))

    return SweepResult(
        results=results,
        param_combinations=combinations,
        metric=metric,
    )
