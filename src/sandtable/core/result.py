"""
src/sandtable/core/result.py

BacktestResult container for backtest outputs.

Provides convenient accessors, DataFrame conversions, and delegates
to visualization and report modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from sandtable.core.events import FillEvent
from sandtable.metrics.performance import PerformanceMetrics
from sandtable.portfolio.portfolio import EquityPoint
from sandtable.report.tearsheet import generate_tearsheet
from sandtable.viz.charts import plot_backtest_results


@dataclass(frozen=True)
class BacktestResult:
    """
    Immutable container for all backtest outputs.

    Attributes:
        metrics: Calculated performance metrics
        equity_curve: List of equity snapshots over time
        trades: List of all executed fill events
        parameters: Dict of all config used for this run
        price_data: Dict mapping symbol to DataFrame of OHLCV data
        symbols: List of symbols traded
        initial_capital: Starting capital
        start_date: First bar date
        end_date: Last bar date
    """

    metrics: PerformanceMetrics
    equity_curve: list[EquityPoint]
    trades: list[FillEvent]
    parameters: dict[str, Any]
    price_data: dict[str, pd.DataFrame]
    symbols: list[str]
    initial_capital: float
    start_date: datetime | None
    end_date: datetime | None

    def __str__(self) -> str:
        return (
            f"BacktestResult(\n"
            f"  symbols={self.symbols},\n"
            f"  period={self.start_date} to {self.end_date},\n"
            f"  {self.metrics}\n"
            f")"
        )

    def equity_dataframe(self) -> pd.DataFrame:
        """Convert equity curve to a DataFrame."""
        return pd.DataFrame(
            [
                {
                    "timestamp": p.timestamp,
                    "equity": p.equity,
                    "cash": p.cash,
                    "positions_value": p.positions_value,
                }
                for p in self.equity_curve
            ]
        )

    def trades_dataframe(self) -> pd.DataFrame:
        """Convert trades to a DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(
            data=[
                {
                    "timestamp": t.timestamp,
                    "symbol": t.symbol,
                    "direction": t.direction.name,
                    "quantity": t.quantity,
                    "fill_price": t.fill_price,
                    "commission": t.commission,
                    "slippage": t.slippage,
                    "market_impact": t.market_impact,
                }
                for t in self.trades
            ]
        )

    def plot(self, **kwargs: Any) -> Any:
        """Plot backtest results using matplotlib."""
        return plot_backtest_results(
            equity_curve=self.equity_curve,
            trades=self.trades,
            price_data=self.price_data.get(self.symbols[0]) if self.symbols else None,
            metrics=self.metrics,
            **kwargs,
        )

    def tearsheet(self, output_path: str | None = None) -> str:
        """Generate an HTML tearsheet report."""
        return generate_tearsheet(self, output_path=output_path)
