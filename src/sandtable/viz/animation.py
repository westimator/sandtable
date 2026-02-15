"""
src/sandtable/viz/animation.py

Animated replay of backtest execution.
"""

from dataclasses import dataclass
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

from sandtable.core.events import Direction, FillEvent
from sandtable.portfolio.portfolio import EquityPoint


@dataclass
class AnimationState:
    """Tracks state for the animation."""

    current_frame: int = 0
    buy_times: list = None
    buy_prices: list = None
    sell_times: list = None
    sell_prices: list = None

    def __post_init__(self):
        self.buy_times = []
        self.buy_prices = []
        self.sell_times = []
        self.sell_prices = []


def animate_backtest(
    equity_curve: list[EquityPoint],
    trades: list[FillEvent],
    price_data: pd.DataFrame,
    interval: int = 50,
    title: str = "Backtest replay",
    figsize: tuple[int, int] = (14, 8),
    save_path: str | None = None,
) -> FuncAnimation:
    """
    Create an animated replay of the backtest.

    Watches the backtest unfold bar-by-bar with:
    - Price chart building up over time
    - Buy/sell markers appearing when trades happen
    - Equity curve growing
    - Real-time P&L display

    Args:
        equity_curve: List of EquityPoint from portfolio
        trades: List of FillEvent from portfolio
        price_data: DataFrame with 'timestamp' and 'close' columns
        interval: Milliseconds between frames (lower = faster)
        title: Animation title
        figsize: Figure size (width, height)
        save_path: Optional path to save as .mp4 or .gif

    Returns:
        FuncAnimation object (call plt.show() to display)
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax_price = axes[0]
    ax_equity = axes[1]

    # Prepare data
    timestamps = price_data["timestamp"].tolist()
    prices = price_data["close"].tolist()
    eq_timestamps = [p.timestamp for p in equity_curve]
    eq_values = [p.equity for p in equity_curve]
    initial_equity = eq_values[0] if eq_values else 100000

    # Build trade lookup by timestamp
    trade_lookup: dict[datetime, list[FillEvent]] = {}
    for trade in trades:
        if trade.timestamp not in trade_lookup:
            trade_lookup[trade.timestamp] = []
        trade_lookup[trade.timestamp].append(trade)

    # Set up axes limits
    ax_price.set_xlim(timestamps[0], timestamps[-1])
    ax_price.set_ylim(min(prices) * 0.98, max(prices) * 1.02)
    ax_equity.set_xlim(eq_timestamps[0], eq_timestamps[-1])
    ax_equity.set_ylim(min(eq_values) * 0.98, max(eq_values) * 1.02)

    # Initialize plot elements
    price_line, = ax_price.plot([], [], color="steelblue", linewidth=1.5, label="Price")
    buy_scatter = ax_price.scatter([], [], marker="^", color="green", s=120, label="Buy", zorder=5)
    sell_scatter = ax_price.scatter([], [], marker="v", color="red", s=120, label="Sell", zorder=5)

    equity_line, = ax_equity.plot([], [], color="darkgreen", linewidth=2, label="Equity")
    _initial_line = ax_equity.axhline(y=initial_equity, color="gray", linestyle="--", alpha=0.5, label="Initial")

    # Labels and legends
    ax_price.set_ylabel("Price ($)")
    ax_price.legend(loc="lower left")
    ax_price.grid(True, alpha=0.3)
    ax_price.set_title("Price and trades")

    ax_equity.set_ylabel("Equity ($)")
    ax_equity.set_xlabel("Date")
    ax_equity.legend(loc="upper left")
    ax_equity.grid(True, alpha=0.3)
    ax_equity.set_title("Portfolio equity")
    ax_equity.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    # Info text
    info_text = ax_price.text(
        0.02, 0.95, "", transform=ax_price.transAxes,
        fontsize=10, fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    # Animation state
    state = AnimationState()

    def init():
        """Initialize animation."""
        price_line.set_data([], [])
        equity_line.set_data([], [])
        # Empty 2D array for scatter - matplotlib requires shape (N, 2)
        empty_offsets = np.empty((0, 2))
        buy_scatter.set_offsets(empty_offsets)
        sell_scatter.set_offsets(empty_offsets)
        info_text.set_text("")
        return price_line, equity_line, buy_scatter, sell_scatter, info_text

    def update(frame):
        """Update animation for each frame."""
        # Update price line
        price_line.set_data(timestamps[: frame + 1], prices[: frame + 1])

        # Update equity line
        eq_frame = min(frame, len(eq_values) - 1)
        equity_line.set_data(eq_timestamps[: eq_frame + 1], eq_values[: eq_frame + 1])

        # Check for trades at this timestamp
        current_time = timestamps[frame]
        if current_time in trade_lookup:
            for trade in trade_lookup[current_time]:
                if trade.direction == Direction.LONG:
                    state.buy_times.append(trade.timestamp)
                    state.buy_prices.append(trade.fill_price)
                else:
                    state.sell_times.append(trade.timestamp)
                    state.sell_prices.append(trade.fill_price)

        # Update scatter plots (convert datetimes to matplotlib date numbers)
        if state.buy_times:
            buy_dates = mdates.date2num(state.buy_times)
            buy_scatter.set_offsets(np.column_stack([buy_dates, state.buy_prices]))
        if state.sell_times:
            sell_dates = mdates.date2num(state.sell_times)
            sell_scatter.set_offsets(np.column_stack([sell_dates, state.sell_prices]))

        # Update info text
        current_equity = eq_values[eq_frame]
        pnl = current_equity - initial_equity
        pnl_pct = (pnl / initial_equity) * 100
        num_trades = len(state.buy_times) + len(state.sell_times)

        info_text.set_text(
            f"Date: {current_time.strftime('%Y-%m-%d')}\n"
            f"Price: ${prices[frame]:.2f}\n"
            f"Equity: ${current_equity:,.0f}\n"
            f"P&L: ${pnl:+,.0f} ({pnl_pct:+.2f}%)\n"
            f"Trades: {num_trades}"
        )

        return price_line, equity_line, buy_scatter, sell_scatter, info_text

    # Create animation
    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(timestamps),
        interval=interval,
        blit=True,
        repeat=False,
    )

    plt.tight_layout()

    if save_path:
        if save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=1000 // interval)
        else:
            anim.save(save_path, writer="ffmpeg", fps=1000 // interval)

    return anim
