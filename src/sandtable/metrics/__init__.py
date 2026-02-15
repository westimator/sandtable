"""
sandtable.metrics - Performance metrics and optimization targets.
"""

from enum import StrEnum

from sandtable.metrics.performance import PerformanceMetrics


class Metric(StrEnum):
    """Metric names that can be used as optimization targets."""

    TOTAL_RETURN = "total_return"
    CAGR = "cagr"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    AVG_TRADE_PNL = "avg_trade_pnl"


__all__ = ["Metric", "PerformanceMetrics"]
