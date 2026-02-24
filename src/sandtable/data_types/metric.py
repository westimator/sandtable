"""
src/sandtable/data_types/metric.py

Metric names that can be used as optimization targets.
"""

from __future__ import annotations

from enum import StrEnum


class Metric(StrEnum):
    """
    Metric names that can be used as optimization targets.

    Values:
        TOTAL_RETURN: cumulative percentage return over the full period
        CAGR: compound annual growth rate
        SHARPE_RATIO: annualized risk-adjusted return (excess return / volatility)
        SORTINO_RATIO: like Sharpe but penalizes only downside volatility
        MAX_DRAWDOWN: largest peak-to-trough decline as a fraction
        WIN_RATE: fraction of round-trip trades that were profitable
        PROFIT_FACTOR: gross profit / gross loss
        AVG_TRADE_PNL: mean P&L per round-trip trade in currency units
    """

    TOTAL_RETURN = "total_return"
    CAGR = "cagr"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    AVG_TRADE_PNL = "avg_trade_pnl"
