"""
src/sandtable/viz/charts.py

Static chart visualizations for backtest results.
"""


import matplotlib.pyplot as plt
import pandas as pd

from sandtable.core.events import Direction, FillEvent
from sandtable.metrics.performance import PerformanceMetrics
from sandtable.portfolio.portfolio import EquityPoint


def plot_backtest_results(
    equity_curve: list[EquityPoint],
    trades: list[FillEvent],
    price_data: pd.DataFrame,
    metrics: PerformanceMetrics | None = None,
    title: str = "Backtest results",
    figsize: tuple[int, int] = (14, 10),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Create a multi-panel chart showing backtest results.

    Panel 1: Price chart with buy/sell markers
    Panel 2: Equity curve
    Panel 3: Drawdown

    Args:
        equity_curve: List of EquityPoint from portfolio
        trades: List of FillEvent from portfolio
        price_data: DataFrame with 'timestamp' and 'close' columns
        metrics: Optional PerformanceMetrics to display
        title: Chart title
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Extract data
    eq_timestamps = [p.timestamp for p in equity_curve]
    eq_values = [p.equity for p in equity_curve]

    # Panel 1: Price with trade markers
    ax1 = axes[0]
    timestamps = price_data["timestamp"] if "timestamp" in price_data.columns else price_data.index
    ax1.plot(timestamps, price_data["close"], label="Price", color="steelblue", linewidth=1)

    # Plot buy/sell markers
    buys = [t for t in trades if t.direction == Direction.LONG]
    sells = [t for t in trades if t.direction == Direction.SHORT]

    if buys:
        buy_times = [t.timestamp for t in buys]
        buy_prices = [t.fill_price for t in buys]
        ax1.scatter(buy_times, buy_prices, marker="^", color="green", s=100, label="Buy", zorder=5)

    if sells:
        sell_times = [t.timestamp for t in sells]
        sell_prices = [t.fill_price for t in sells]
        ax1.scatter(sell_times, sell_prices, marker="v", color="red", s=100, label="Sell", zorder=5)

    ax1.set_ylabel("Price ($)")
    ax1.legend(loc="lower left")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Price and trades")

    # Panel 2: Equity curve
    ax2 = axes[1]
    ax2.plot(eq_timestamps, eq_values, label="Equity", color="darkgreen", linewidth=1.5)
    ax2.axhline(y=eq_values[0], color="gray", linestyle="--", alpha=0.5, label="Initial capital")
    ax2.fill_between(eq_timestamps, eq_values[0], eq_values, alpha=0.3, color="green")
    ax2.set_ylabel("Equity ($)")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Portfolio equity")

    # Format y-axis with commas
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    # Panel 3: Drawdown
    ax3 = axes[2]
    eq_series = pd.Series(eq_values, index=eq_timestamps)
    rolling_max = eq_series.expanding().max()
    drawdown = (eq_series - rolling_max) / rolling_max * 100

    ax3.fill_between(eq_timestamps, 0, drawdown, color="red", alpha=0.4)
    ax3.plot(eq_timestamps, drawdown, color="darkred", linewidth=1)
    ax3.set_ylabel("Drawdown (%)")
    ax3.set_xlabel("Date")
    ax3.grid(True, alpha=0.3)
    ax3.set_title("Drawdown")

    # Add metrics annotation if provided
    if metrics is not None:
        metrics_text = (
            f"Total return: {metrics.total_return * 100:+.2f}%\n"
            f"Sharpe ratio: {metrics.sharpe_ratio:.2f}\n"
            f"Max drawdown: {metrics.max_drawdown * 100:.2f}%\n"
            f"Win rate: {metrics.win_rate * 100:.1f}%\n"
            f"Trades: {metrics.num_trades}"
        )
        fig.text(
            0.02, 0.02, metrics_text,
            fontsize=9, fontfamily="monospace",
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
