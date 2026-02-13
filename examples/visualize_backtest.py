"""
examples/visualize_backtest.py

Example: Visualize backtest results with charts and animation.

This example demonstrates:
- Running a backtest
- Creating static charts (price + equity + drawdown)
- Running an animated replay

Usage:
    # Static chart only:
    uv run python examples/visualize_backtest.py

    # With animation:
    uv run python examples/visualize_backtest.py --animate
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from backtester.core.backtest import Backtest
from backtester.data.data_handler import CSVDataHandler
from backtester.execution.simulator import ExecutionConfig, ExecutionSimulator
from backtester.execution.slippage import FixedSlippage
from backtester.portfolio.portfolio import Portfolio
from backtester.strategy.ma_crossover import MACrossoverStrategy
from backtester.viz import plot_backtest_results, animate_backtest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_data_file() -> Path:
    """Find the sample data file."""
    script_dir = Path(__file__).parent
    data_file = script_dir.parent / "data" / "sample_ohlcv.csv"
    if data_file.exists():
        return data_file
    cwd_data = Path("data/sample_ohlcv.csv")
    if cwd_data.exists():
        return cwd_data
    raise FileNotFoundError("Could not find data/sample_ohlcv.csv")


def run_backtest():
    """Run the backtest and return results."""
    data_file = find_data_file()
    logger.info("Loading data from: %s", data_file)

    data = CSVDataHandler(data_file, "SPY")
    strategy = MACrossoverStrategy(fast_period=10, slow_period=30)
    portfolio = Portfolio(initial_capital=100_000)
    executor = ExecutionSimulator(
        config=ExecutionConfig(commission_per_share=0.005),
        slippage_model=FixedSlippage(bps=5),
    )

    backtest = Backtest(data, strategy, portfolio, executor)
    logger.info("Running backtest...")
    metrics = backtest.run()

    return data, portfolio, metrics


def main():
    parser = argparse.ArgumentParser(description="Visualize backtest results")
    parser.add_argument("--animate", action="store_true", help="Show animated replay")
    parser.add_argument("--speed", type=int, default=50, help="Animation speed (ms per frame)")
    args = parser.parse_args()

    # Run backtest
    data, portfolio, metrics = run_backtest()

    # Print metrics
    logger.info("=" * 50)
    logger.info("RESULTS")
    logger.info("=" * 50)
    logger.info("Total Return: %+.2f%%", metrics.total_return * 100)
    logger.info("Sharpe Ratio: %.2f", metrics.sharpe_ratio)
    logger.info("Max Drawdown: %.2f%%", metrics.max_drawdown * 100)
    logger.info("Trades: %d", metrics.num_trades)
    logger.info("Win Rate: %.1f%%", metrics.win_rate * 100)

    # Get price data
    price_data = data.data[["timestamp", "close"]]

    # Create static chart
    logger.info("Creating static chart...")
    fig = plot_backtest_results(
        equity_curve=portfolio.equity_curve,
        trades=portfolio.trades,
        price_data=price_data,
        metrics=metrics,
        title="MA crossover strategy on SPY",
        save_path="backtest_results.png",
    )

    logger.info("Saved chart to backtest_results.png")

    if args.animate:
        logger.info("Starting animated replay (close window to exit)...")
        anim = animate_backtest(
            equity_curve=portfolio.equity_curve,
            trades=portfolio.trades,
            price_data=price_data,
            interval=args.speed,
            title="MA crossover strategy live replay",
            save_path="backtest_animation.gif",
        )
        logger.info("Saved animation to backtest_animation.gif")

    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
