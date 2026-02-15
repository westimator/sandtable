"""
examples/advanced/visualize_backtest.py

Advanced example: visualize backtest results with charts and animation.
Uses manual component wiring for full control.

Usage:
    uv run python examples/advanced/visualize_backtest.py
    uv run python examples/advanced/visualize_backtest.py --animate
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from sandtable import CSVDataHandler, MACrossoverStrategy, settings
from sandtable.core import Backtest
from sandtable.execution import ExecutionConfig, ExecutionSimulator, FixedSlippage
from sandtable.portfolio import Portfolio
from sandtable.viz import animate_backtest, plot_backtest_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def do_backtest():
    """Run the backtest and return results."""
    data_file = DATA_DIR / "sample_ohlcv.csv"
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

    data, portfolio, metrics = do_backtest()

    logger.info("=" * 50)
    logger.info("RESULTS")
    logger.info("=" * 50)
    logger.info("Total Return: %+.2f%%", metrics.total_return * 100)
    logger.info("Sharpe Ratio: %.2f", metrics.sharpe_ratio)
    logger.info("Max Drawdown: %.2f%%", metrics.max_drawdown * 100)
    logger.info("Trades: %d", metrics.num_trades)
    logger.info("Win Rate: %.1f%%", metrics.win_rate * 100)

    price_data = data.data[["timestamp", "close"]]

    output_dir = settings.BACKTESTER_OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)

    logger.info("Creating static chart...")
    _fig = plot_backtest_results(
        equity_curve=portfolio.equity_curve,
        trades=portfolio.trades,
        price_data=price_data,
        metrics=metrics,
        title="MA crossover strategy on SPY",
        save_path=str(output_dir / "backtest_results.png"),
    )
    logger.info("Saved chart to %s", output_dir / "backtest_results.png")

    if args.animate:
        logger.info("Starting animated replay (close window to exit)...")
        _anim = animate_backtest(
            equity_curve=portfolio.equity_curve,
            trades=portfolio.trades,
            price_data=price_data,
            interval=args.speed,
            title="MA crossover strategy live replay",
            save_path=str(output_dir / "backtest_animation.gif"),
        )
        logger.info("Saved animation to %s", output_dir / "backtest_animation.gif")

    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
