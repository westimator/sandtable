"""
src/sandtable/research/walkforward.py

Walk-forward analysis - the only honest way to evaluate a trading strategy.

Splits data into train/test folds, optimizes parameters in-sample on each
train window, then evaluates out-of-sample on the subsequent test window.
The stitched OOS equity curve is the only non-overfit curve the system produces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sandtable.api import SweepResult, run_backtest, run_parameter_sweep
from sandtable.core.result import BacktestResult
from sandtable.data_engine.handler import DataHandler
from sandtable.data_types.metric import Metric
from sandtable.metrics.performance import (
    calculate_cagr,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
)
from sandtable.portfolio.portfolio import EquityPoint
from sandtable.strategy.abstract_strategy import AbstractStrategy
from sandtable.utils.exceptions import InsufficientDataError
from sandtable.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class WalkForwardFold:
    """Results for a single walk-forward fold."""

    fold_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    chosen_params: dict[str, Any]
    in_sample_metrics: dict[str, float]
    out_of_sample_metrics: dict[str, float]
    equity_curve: list[EquityPoint]


@dataclass
class WalkForwardResult:
    """Aggregate results from a walk-forward analysis."""

    folds: list[WalkForwardFold]
    oos_equity_curve: list[EquityPoint] = field(default_factory=list)
    oos_sharpe: float = 0.0
    oos_max_drawdown: float = 0.0
    oos_cagr: float = 0.0
    config: dict[str, Any] = field(default_factory=dict)


def _get_trading_dates(data: DataHandler) -> list[datetime]:
    """Extract sorted unique trading dates from all loaded data."""
    dates: set[datetime] = set()
    for df in data.data.values():
        dates.update(df.index.to_pydatetime())
    return sorted(dates)


def _metrics_to_dict(result: BacktestResult) -> dict[str, float]:
    """Extract key metrics as a flat dict."""
    m = result.metrics
    return {
        Metric.SHARPE_RATIO: m.sharpe_ratio,
        Metric.SORTINO_RATIO: m.sortino_ratio,
        Metric.MAX_DRAWDOWN: m.max_drawdown,
        Metric.CAGR: m.cagr,
        Metric.TOTAL_RETURN: m.total_return,
        Metric.WIN_RATE: m.win_rate,
        Metric.PROFIT_FACTOR: m.profit_factor,
        Metric.AVG_TRADE_PNL: m.avg_trade_pnl,
        "num_trades": m.num_trades,
    }


def run_walkforward(
    strategy_cls: type[AbstractStrategy],
    param_grid: dict[str, list[Any]],
    data: DataHandler,
    train_window: int,
    test_window: int,
    step_size: int | None = None,
    optimization_metric: Metric = Metric.SHARPE_RATIO,
    **backtest_kwargs: Any,
) -> WalkForwardResult:
    """
    Run walk-forward analysis with rolling train/test folds.

    Args:
        strategy_cls: Strategy class to optimize
        param_grid: Parameter grid for sweep on each train window
        data: DataHandler with ALL data pre-loaded
        train_window: Number of trading days for in-sample training
        test_window: Number of trading days for out-of-sample testing
        step_size: Days to advance between folds (defaults to test_window)
        optimization_metric: Metric to optimize during train sweep
        **backtest_kwargs: Additional kwargs for run_backtest (e.g. commission, slippage)

    Returns:
        WalkForwardResult with per-fold detail and stitched OOS curve
    """
    if step_size is None:
        step_size = test_window

    trading_dates = _get_trading_dates(data)
    total_dates = len(trading_dates)

    if total_dates < train_window + test_window:
        raise InsufficientDataError(
            f"Insufficient data: {total_dates} trading days available, "
            f"need at least {train_window + test_window} (train + test)"
        )

    # generate fold boundaries
    folds: list[WalkForwardFold] = []
    oos_equity_points: list[EquityPoint] = []
    fold_idx = 0
    start = 0

    while start + train_window + test_window <= total_dates:
        train_start_date = trading_dates[start]
        train_end_date = trading_dates[start + train_window - 1]
        test_start_date = trading_dates[start + train_window]
        test_end_idx = min(start + train_window + test_window - 1, total_dates - 1)
        test_end_date = trading_dates[test_end_idx]

        train_start_str = train_start_date.strftime("%Y-%m-%d")
        train_end_str = train_end_date.strftime("%Y-%m-%d")
        test_start_str = test_start_date.strftime("%Y-%m-%d")
        test_end_str = test_end_date.strftime("%Y-%m-%d")

        logger.info(
            "Fold %d: train [%s, %s] test [%s, %s]",
            fold_idx, train_start_str, train_end_str, test_start_str, test_end_str,
        )

        # in-sample: sweep on train window
        train_data = data.date_slice(train_start_str, train_end_str)
        sweep_result: SweepResult = run_parameter_sweep(
            strategy_class=strategy_cls,
            param_grid=param_grid,
            data=train_data,
            metric=optimization_metric,
            **backtest_kwargs,
        )

        chosen_params = sweep_result.best_params
        is_metrics = _metrics_to_dict(sweep_result.best_result)

        logger.info(
            "Fold %d: best params = %s (IS %s=%.4f)",
            fold_idx, chosen_params, optimization_metric, is_metrics.get(optimization_metric, 0.0),
        )

        # out-of-sample: single run with chosen params
        test_data = data.date_slice(
            start_date=test_start_str,
            end_date=test_end_str,
        )
        strategy = strategy_cls(**chosen_params)
        oos_result = run_backtest(
            strategy=strategy,
            data=test_data,
            **backtest_kwargs,
        )
        oos_metrics = _metrics_to_dict(oos_result)

        fold = WalkForwardFold(
            fold_index=fold_idx,
            train_start=train_start_str,
            train_end=train_end_str,
            test_start=test_start_str,
            test_end=test_end_str,
            chosen_params=chosen_params,
            in_sample_metrics=is_metrics,
            out_of_sample_metrics=oos_metrics,
            equity_curve=oos_result.equity_curve,
        )
        folds.append(fold)
        oos_equity_points.extend(oos_result.equity_curve)

        fold_idx += 1
        start += step_size

    # compute aggregate OOS metrics from stitched curve
    oos_sharpe = 0.0
    oos_max_drawdown = 0.0
    oos_cagr = 0.0

    if oos_equity_points:
        equities = [p.equity for p in oos_equity_points]
        daily_returns = []
        for i in range(1, len(equities)):
            if equities[i - 1] != 0:
                daily_returns.append((equities[i] - equities[i - 1]) / equities[i - 1])

        oos_sharpe = calculate_sharpe_ratio(daily_returns)
        oos_max_drawdown = calculate_max_drawdown(equities)
        oos_cagr = calculate_cagr(
            start_equity=equities[0],
            end_equity=equities[-1],
            num_days=len(equities),
        )

    return WalkForwardResult(
        folds=folds,
        oos_equity_curve=oos_equity_points,
        oos_sharpe=oos_sharpe,
        oos_max_drawdown=oos_max_drawdown,
        oos_cagr=oos_cagr,
        config={
            "strategy_cls": strategy_cls.__name__,
            "param_grid": param_grid,
            "train_window": train_window,
            "test_window": test_window,
            "step_size": step_size,
            "optimization_metric": str(optimization_metric),
        },
    )
