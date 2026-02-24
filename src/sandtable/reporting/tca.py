"""
src/sandtable/reporting/tca.py

Transaction cost analysis (TCA) aggregation.

Decomposes fill-level costs (slippage, market impact, commission)
into summary statistics by component and by symbol.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sandtable.core.events import FillEvent


@dataclass
class TCAReport:
    """
    Aggregated transaction cost analysis.

    Attributes:
        total_slippage: Total slippage cost across all fills
        total_impact: Total market impact cost across all fills
        total_commission: Total commission cost across all fills
        total_cost: Sum of all cost components
        gross_pnl: PnL before transaction costs
        net_pnl: PnL after transaction costs
        cost_per_trade: Average total cost per fill
        cost_as_pct_of_gross: Total cost as percentage of absolute gross PnL
        cost_by_symbol: Per-symbol cost breakdown
        cost_by_component: Cost totals keyed by component name
    """
    total_slippage: float
    total_impact: float
    total_commission: float
    total_cost: float
    gross_pnl: float
    net_pnl: float
    cost_per_trade: float
    cost_as_pct_of_gross: float
    cost_by_symbol: dict[str, dict[str, float]] = field(default_factory=dict)
    cost_by_component: dict[str, float] = field(default_factory=dict)


def compute_tca(fills: list[FillEvent], gross_pnl: float) -> TCAReport:
    """
    Aggregate fill-level costs into a TCA summary.

    Args:
        fills: List of FillEvent objects from the backtest
        gross_pnl: Gross PnL (before costs) for percentage calculation

    Returns:
        TCAReport with cost decomposition
    """
    if not fills:
        return TCAReport(
            total_slippage=0.0,
            total_impact=0.0,
            total_commission=0.0,
            total_cost=0.0,
            gross_pnl=gross_pnl,
            net_pnl=gross_pnl,
            cost_per_trade=0.0,
            cost_as_pct_of_gross=0.0,
            cost_by_symbol={},
            cost_by_component={
                "slippage": 0.0,
                "impact": 0.0,
                "commission": 0.0,
            },
        )

    total_slippage = 0.0
    total_impact = 0.0
    total_commission = 0.0
    by_symbol: dict[str, dict[str, float]] = {}

    for fill in fills:
        total_slippage += fill.slippage
        total_impact += fill.market_impact
        total_commission += fill.commission

        if fill.symbol not in by_symbol:
            by_symbol[fill.symbol] = {
                "slippage": 0.0,
                "impact": 0.0,
                "commission": 0.0,
                "total": 0.0,
            }

        by_symbol[fill.symbol]["slippage"] += fill.slippage
        by_symbol[fill.symbol]["impact"] += fill.market_impact
        by_symbol[fill.symbol]["commission"] += fill.commission
        by_symbol[fill.symbol]["total"] += fill.total_cost

    total_cost = total_slippage + total_impact + total_commission
    net_pnl = gross_pnl - total_cost
    cost_per_trade = total_cost / len(fills)
    abs_gross = abs(gross_pnl)
    cost_as_pct = (total_cost / abs_gross * 100) if abs_gross > 0 else 0.0

    return TCAReport(
        total_slippage=total_slippage,
        total_impact=total_impact,
        total_commission=total_commission,
        total_cost=total_cost,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        cost_per_trade=cost_per_trade,
        cost_as_pct_of_gross=cost_as_pct,
        cost_by_symbol=by_symbol,
        cost_by_component={
            "slippage": total_slippage,
            "impact": total_impact,
            "commission": total_commission,
        },
    )
