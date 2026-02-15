"""
src/sandtable/metrics/performance.py

Performance metrics calculation for backtesting results.

Calculates returns, risk metrics, and trade statistics from
equity curves and trade histories.
"""

import math
from dataclasses import dataclass

from sandtable.config import settings
from sandtable.core.events import Direction, FillEvent
from sandtable.portfolio.portfolio import EquityPoint
from sandtable.utils.cli import BOLD, RESET, color_value
from sandtable.utils.logger import get_logger

logger = get_logger(__name__)

# annualization factors (from config / env vars)
TRADING_DAYS_PER_YEAR = settings.BACKTESTER_TRADING_DAYS
RISK_FREE_RATE = settings.BACKTESTER_RISK_FREE_RATE


@dataclass
class PerformanceMetrics:
    """
    Container for backtest performance metrics.

    Attributes:
        # Return metrics
        total_return: Total return as decimal (0.10 = 10%)
        cagr: Compound annual growth rate as decimal

        # Risk metrics
        sharpe_ratio: Annualized Sharpe ratio (excess return / volatility)
        sortino_ratio: Annualized Sortino ratio (excess return / downside volatility)
        max_drawdown: Maximum peak-to-trough decline as decimal

        # Trade metrics
        num_trades: Total number of round-trip trades
        win_rate: Fraction of winning trades (0.0 to 1.0)
        profit_factor: Gross profits / gross losses
        avg_trade_pnl: Average P&L per trade

        # Summary
        start_equity: Starting portfolio equity
        end_equity: Ending portfolio equity
        num_days: Number of trading days in backtest
    """

    # return metrics
    total_return: float
    cagr: float

    # risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float

    # trade metrics
    num_fills: int
    num_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float

    # summary
    start_equity: float
    end_equity: float
    num_days: int

    def __str__(self) -> str:
        """
        Format metrics for display with ANSI colors.
        """
        cv = color_value
        return (
            f"PerformanceMetrics(\n"
            f"  {BOLD}Returns:{RESET}\n"
            f"    total_return:  {cv(self.total_return, f'{self.total_return:+.2%}')}\n"
            f"    cagr:          {cv(self.cagr, f'{self.cagr:+.2%}')}\n"
            f"  {BOLD}Risk:{RESET}\n"
            f"    sharpe_ratio:  {cv(self.sharpe_ratio, f'{self.sharpe_ratio:.2f}')}\n"
            f"    sortino_ratio: {cv(self.sortino_ratio, f'{self.sortino_ratio:.2f}')}\n"
            f"    max_drawdown:  {cv(self.max_drawdown, f'{self.max_drawdown:.2%}', invert=True)}\n"
            f"  {BOLD}Trades:{RESET}\n"
            f"    num_fills:     {self.num_fills}\n"
            f"    num_trades:    {self.num_trades}\n"
            f"    win_rate:      {self.win_rate:.1%}\n"
            f"    profit_factor: {cv(self.profit_factor - 1, f'{self.profit_factor:.2f}')}\n"
            f"    avg_trade_pnl: {cv(self.avg_trade_pnl, f'${self.avg_trade_pnl:.2f}')}\n"
            f"  {BOLD}Summary:{RESET}\n"
            f"    start_equity:  ${self.start_equity:,.2f}\n"
            f"    end_equity:    ${self.end_equity:,.2f}\n"
            f"    num_days:      {self.num_days}\n"
            f")"
        )


def calculate_metrics(
    equity_curve: list[EquityPoint],
    trades: list[FillEvent],
    risk_free_rate: float = RISK_FREE_RATE,
) -> PerformanceMetrics:
    """
    Calculate performance metrics from equity curve and trades.

    Args:
        equity_curve: List of equity snapshots over time
        trades: List of all executed fills
        risk_free_rate: Annual risk-free rate for Sharpe/Sortino (default 0)

    Returns:
        PerformanceMetrics with all calculated values
    """
    logger.debug("Calculating metrics from %d equity points and %d trades", len(equity_curve), len(trades))

    if len(equity_curve) < 2:
        logger.warning("Insufficient equity curve data for metrics calculation")
        return _empty_metrics()

    # Extract equity values
    equities = [point.equity for point in equity_curve]
    start_equity = equities[0]
    end_equity = equities[-1]
    num_days = len(equity_curve)

    # calculate returns
    total_return = (end_equity - start_equity) / start_equity
    cagr = _calculate_cagr(start_equity, end_equity, num_days)

    # calculate daily returns for risk metrics
    daily_returns = _calculate_daily_returns(equities)

    # risk metrics
    sharpe_ratio = _calculate_sharpe_ratio(daily_returns, risk_free_rate)
    sortino_ratio = _calculate_sortino_ratio(daily_returns, risk_free_rate)
    max_drawdown = _calculate_max_drawdown(equities)

    # trade metrics
    trade_metrics = _calculate_trade_metrics(trades)

    metrics = PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown=max_drawdown,
        num_fills=len(trades),
        num_trades=trade_metrics["num_trades"],
        win_rate=trade_metrics["win_rate"],
        profit_factor=trade_metrics["profit_factor"],
        avg_trade_pnl=trade_metrics["avg_trade_pnl"],
        start_equity=start_equity,
        end_equity=end_equity,
        num_days=num_days,
    )

    logger.debug("Performance metrics calculated: %s", metrics)
    return metrics


def _empty_metrics() -> PerformanceMetrics:
    """
    Return empty metrics when calculation isn't possible.
    """
    return PerformanceMetrics(
        total_return=0.0,
        cagr=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        max_drawdown=0.0,
        num_fills=0,
        num_trades=0,
        win_rate=0.0,
        profit_factor=0.0,
        avg_trade_pnl=0.0,
        start_equity=0.0,
        end_equity=0.0,
        num_days=0,
    )


def _calculate_daily_returns(equities: list[float]) -> list[float]:
    """
    Calculate daily returns from equity series.
    """
    returns = []
    for i in range(1, len(equities)):
        if equities[i - 1] != 0:
            daily_return = (equities[i] - equities[i - 1]) / equities[i - 1]
            returns.append(daily_return)
    return returns


def _calculate_cagr(
    start_equity: float,
    end_equity: float,
    num_days: int,
) -> float:
    """
    Calculate compound annual growth rate.

    CAGR = (end_value / start_value) ^ (252 / num_days) - 1
    """
    if start_equity <= 0 or num_days <= 0:
        return 0.0

    if end_equity <= 0:
        return -1.0  # total loss

    years = num_days / TRADING_DAYS_PER_YEAR
    if years <= 0:
        return 0.0

    return (end_equity / start_equity) ** (1 / years) - 1


def _calculate_sharpe_ratio(
    daily_returns: list[float],
    risk_free_rate: float = 0.0,
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Sharpe = (mean_return - risk_free_rate) / std_dev * sqrt(252)
    """
    if len(daily_returns) < 2:
        return 0.0

    mean_return = sum(daily_returns) / len(daily_returns)
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR

    # calculate standard deviation
    variance = sum((r - mean_return) ** 2 for r in daily_returns) / (
        len(daily_returns) - 1
    )
    std_dev = math.sqrt(variance) if variance > 0 else 0.0

    if std_dev == 0:
        return 0.0

    # annualize
    sharpe = (mean_return - daily_rf) / std_dev * math.sqrt(TRADING_DAYS_PER_YEAR)
    return sharpe


def _calculate_sortino_ratio(
    daily_returns: list[float],
    risk_free_rate: float = 0.0,
) -> float:
    """
    Calculate annualized Sortino ratio.

    Sortino = (mean_return - risk_free_rate) / downside_std * sqrt(252)

    Uses only negative returns for downside deviation.
    """
    if len(daily_returns) < 2:
        return 0.0

    mean_return = sum(daily_returns) / len(daily_returns)
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR

    # calculate downside deviation (only negative returns)
    negative_returns = [r for r in daily_returns if r < 0]

    if len(negative_returns) < 2:
        # no downside volatility; return high value if positive returns
        return 10.0 if mean_return > daily_rf else 0.0

    downside_variance = sum(r**2 for r in negative_returns) / (
        len(negative_returns) - 1
    )
    downside_std = math.sqrt(downside_variance) if downside_variance > 0 else 0.0

    if downside_std == 0:
        return 0.0

    # annualize
    sortino = (mean_return - daily_rf) / downside_std * math.sqrt(TRADING_DAYS_PER_YEAR)
    return sortino


def _calculate_max_drawdown(equities: list[float]) -> float:
    """
    Calculate maximum drawdown (peak-to-trough decline).

    Returns as a positive decimal (0.20 = 20% drawdown).
    """
    if len(equities) < 2:
        return 0.0

    max_drawdown = 0.0
    peak = equities[0]

    for equity in equities:
        if equity > peak:
            peak = equity
        elif peak > 0:
            drawdown = (peak - equity) / peak
            max_drawdown = max(max_drawdown, drawdown)

    return max_drawdown


def _calculate_trade_metrics(trades: list[FillEvent]) -> dict:
    """
    Calculate trade-based metrics from fill events.

    Groups fills into round-trip trades and calculates statistics.
    """
    if not trades:
        return {
            "num_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_trade_pnl": 0.0,
        }

    # group trades by symbol and calculate P&L for round trips;
    # a round trip is: entry (buy/sell) -> exit (opposite direction)
    round_trips = _extract_round_trips(trades)

    if not round_trips:
        return {
            "num_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_trade_pnl": 0.0,
        }

    # calculate metrics from round trips
    num_trades = len(round_trips)
    pnls = [rt["pnl"] for rt in round_trips]

    winning_trades = [p for p in pnls if p > 0]
    losing_trades = [p for p in pnls if p < 0]

    win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0.0

    gross_profit = sum(winning_trades) if winning_trades else 0.0
    gross_loss = abs(sum(losing_trades)) if losing_trades else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # cap profit factor at a reasonable value for display
    if profit_factor == float("inf"):
        profit_factor = 999.99

    avg_trade_pnl = sum(pnls) / num_trades if num_trades > 0 else 0.0

    return {
        "num_trades": num_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_trade_pnl": avg_trade_pnl,
    }


def _extract_round_trips(trades: list[FillEvent]) -> list[dict]:
    """
    Extract round-trip trades from fill events.

    A round trip consists of an entry and exit. Handles partial fills
    and position flipping by tracking running positions.

    Returns list of dicts with 'entry_price', 'exit_price', 'quantity', 'pnl'.
    """
    round_trips = []
    positions: dict[str, list[dict]] = {}  # symbol -> list of open positions

    for fill in trades:
        symbol = fill.symbol
        if symbol not in positions:
            positions[symbol] = []

        if fill.direction == Direction.LONG:
            # buying: either opening long or covering short
            remaining_qty = fill.quantity

            # first, close any short positions
            while remaining_qty > 0 and positions[symbol]:
                open_pos = positions[symbol][0]
                if open_pos["direction"] == Direction.SHORT:
                    close_qty = min(remaining_qty, open_pos["quantity"])

                    # p&l for covering short: (entry_price - exit_price) * qty - costs
                    pnl = (open_pos["price"] - fill.fill_price) * close_qty
                    pnl -= fill.commission * (close_qty / fill.quantity)

                    round_trips.append(
                        {
                            "entry_price": open_pos["price"],
                            "exit_price": fill.fill_price,
                            "quantity": close_qty,
                            "pnl": pnl,
                            "direction": "short",
                        }
                    )

                    open_pos["quantity"] -= close_qty
                    remaining_qty -= close_qty

                    if open_pos["quantity"] <= 0:
                        positions[symbol].pop(0)
                else:
                    break

            # any remaining quantity opens a new long position
            if remaining_qty > 0:
                positions[symbol].append(
                    {
                        "direction": Direction.LONG,
                        "price": fill.fill_price,
                        "quantity": remaining_qty,
                        "commission": fill.commission * (remaining_qty / fill.quantity),
                    }
                )

        else:  # SHORT direction
            # selling: either opening short or closing long
            remaining_qty = fill.quantity

            # first, close any long positions
            while remaining_qty > 0 and positions[symbol]:
                open_pos = positions[symbol][0]
                if open_pos["direction"] == Direction.LONG:
                    close_qty = min(remaining_qty, open_pos["quantity"])

                    # p&l for closing long: (exit_price - entry_price) * qty - costs
                    pnl = (fill.fill_price - open_pos["price"]) * close_qty
                    pnl -= fill.commission * (close_qty / fill.quantity)

                    round_trips.append(
                        {
                            "entry_price": open_pos["price"],
                            "exit_price": fill.fill_price,
                            "quantity": close_qty,
                            "pnl": pnl,
                            "direction": "long",
                        }
                    )

                    open_pos["quantity"] -= close_qty
                    remaining_qty -= close_qty

                    if open_pos["quantity"] <= 0:
                        positions[symbol].pop(0)
                else:
                    break

            # any remaining quantity opens a new short position
            if remaining_qty > 0:
                positions[symbol].append(
                    {
                        "direction": Direction.SHORT,
                        "price": fill.fill_price,
                        "quantity": remaining_qty,
                        "commission": fill.commission * (remaining_qty / fill.quantity),
                    }
                )

    return round_trips
