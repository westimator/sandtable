"""
src/sandtable/research/compare.py

Strategy comparison performance table, return correlation, and blended portfolio.

Running a book of strategies and understanding their correlations is
portfolio-level thinking. This module answers: "Do my strategies diversify?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from sandtable.core.result import BacktestResult
from sandtable.data_types.metric import Metric
from sandtable.portfolio.portfolio import EquityPoint
from sandtable.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ComparisonResult:
    """
    Structured output from comparing multiple strategies.

    Attributes:
        results: strategy name -> BacktestResult
        performance_table: rows = strategies, cols = metrics
        correlation_matrix: pairwise Pearson correlation of daily returns
        blended_equity_curve: equal-weight daily rebalance across strategies
    """

    results: dict[str, BacktestResult]
    performance_table: pd.DataFrame
    correlation_matrix: pd.DataFrame
    blended_equity_curve: list[EquityPoint] = field(default_factory=list)


def _daily_returns_series(result: BacktestResult) -> pd.Series:
    """Extract daily returns as a pandas Series indexed by date."""
    eq = result.equity_curve
    if len(eq) < 2:
        return pd.Series(dtype=float)

    dates = [p.timestamp for p in eq]
    equities = [p.equity for p in eq]

    returns = []
    ret_dates = []
    for i in range(1, len(equities)):
        if equities[i - 1] != 0:
            returns.append((equities[i] - equities[i - 1]) / equities[i - 1])
            ret_dates.append(dates[i])

    return pd.Series(
        returns,
        index=pd.DatetimeIndex(ret_dates),
    )


def _build_performance_table(
    results: dict[str, BacktestResult]
) -> pd.DataFrame:
    """Build a DataFrame with one row per strategy and metric columns."""
    rows = []
    for name, result in results.items():
        m = result.metrics
        rows.append({
            "strategy": name,
            Metric.SHARPE_RATIO: m.sharpe_ratio,
            Metric.SORTINO_RATIO: m.sortino_ratio,
            Metric.CAGR: m.cagr,
            Metric.MAX_DRAWDOWN: m.max_drawdown,
            Metric.TOTAL_RETURN: m.total_return,
            Metric.WIN_RATE: m.win_rate,
            Metric.PROFIT_FACTOR: m.profit_factor,
            Metric.AVG_TRADE_PNL: m.avg_trade_pnl,
            "num_trades": m.num_trades,
        })
    return pd.DataFrame(rows).set_index("strategy")


def _build_correlation_matrix(
    return_series: dict[str, pd.Series],
) -> pd.DataFrame:
    """Compute pairwise Pearson correlation of daily returns."""
    df = pd.DataFrame(return_series)
    return df.corr()


def _build_blended_equity(
    results: dict[str, BacktestResult],
) -> list[EquityPoint]:
    """
    Equal-weight daily blend across all strategies.

    For each date, the blended equity is the average of all strategies'
    equity values on that date. This simulates running all strategies
    with equal capital allocation.
    """
    # collect last equity per date per strategy (handles multi-symbol
    # runs where record_equity is called once per bar, not once per day)
    per_strategy: dict[str, dict[datetime, float]] = {}
    for name, result in results.items():
        daily: dict[datetime, float] = {}
        for point in result.equity_curve:
            daily[point.timestamp] = point.equity
        per_strategy[name] = daily

    # only include dates where all strategies have data
    all_dates = set.intersection(*(set(d.keys()) for d in per_strategy.values()))
    blended: list[EquityPoint] = []
    n_strategies = len(results)
    for date in sorted(all_dates):
        avg_equity = sum(d[date] for d in per_strategy.values()) / n_strategies
        blended.append(EquityPoint(
            timestamp=date,
            equity=avg_equity,
            cash=0.0,
            positions_value=avg_equity,
        ))

    return blended


def run_comparison(
    results: dict[str, BacktestResult],
) -> ComparisonResult:
    """
    Compare multiple strategy results side by side.

    Args:
        results: Dict mapping strategy name to its BacktestResult

    Returns:
        ComparisonResult with performance table, correlation matrix,
        and blended equity curve
    """
    if not results:
        raise ValueError("Must provide at least one result to compare")

    logger.info("Comparing %d strategies: %s", len(results), list(results.keys()))

    # performance table
    performance_table = _build_performance_table(results)

    # daily return series for correlation
    return_series = {
        name: _daily_returns_series(r) for name, r in results.items()
    }
    correlation_matrix = _build_correlation_matrix(return_series)

    # blended equity curve
    blended = _build_blended_equity(results)

    return ComparisonResult(
        results=results,
        performance_table=performance_table,
        correlation_matrix=correlation_matrix,
        blended_equity_curve=blended,
    )
