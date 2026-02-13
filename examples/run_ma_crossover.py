"""
examples/run_ma_crossover.py

Example: Run a moving average crossover backtest on SPY data.

This example demonstrates:
- Setting up the backtesting components
- Running a complete backtest
- Viewing performance metrics

Usage:
    uv run python examples/run_ma_crossover.py
    uv run python examples/run_ma_crossover.py --debug
"""

import logging
import sys
from pathlib import Path

from backtester.core.backtest import Backtest
from backtester.data.data_handler import CSVDataHandler
from backtester.execution.simulator import ExecutionConfig, ExecutionSimulator
from backtester.execution.slippage import FixedSlippage, ZeroSlippage
from backtester.portfolio.portfolio import Portfolio
from backtester.strategy.ma_crossover import MACrossoverStrategy

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def find_data_file() -> Path:
    """
    Find the sample data file.
    """
    # try relative to script location
    script_dir = Path(__file__).parent
    data_file = script_dir.parent / "data" / "sample_ohlcv.csv"

    if data_file.exists():
        return data_file

    # try current working directory
    cwd_data = Path("data/sample_ohlcv.csv")
    if cwd_data.exists():
        return cwd_data

    raise FileNotFoundError(
        "Could not find data/sample_ohlcv.csv. Run from the project root directory."
    )


def run_backtest() -> None:
    """
    Run the MA crossover backtest.
    """
    # find data file
    data_file = find_data_file()
    logger.info("Loading data from: %s", data_file)

    # set up components (as shown in project plan)
    data = CSVDataHandler(data_file, "SPY")
    strategy = MACrossoverStrategy(fast_period=10, slow_period=30)
    portfolio = Portfolio(initial_capital=100_000)
    executor = ExecutionSimulator(
        config=ExecutionConfig(commission_per_share=0.005),
        slippage_model=FixedSlippage(bps=5),
    )

    # create and run backtest
    backtest = Backtest(data, strategy, portfolio, executor)

    logger.info("Running backtest on %d bars of %s...", len(data), data.symbol)
    logger.info("Strategy: MA Crossover (fast=%d, slow=%d)", strategy.fast_period, strategy.slow_period)
    logger.info("Initial capital: $%,.2f", portfolio.initial_capital)

    metrics = backtest.run()

    # log results
    logger.info("=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)
    logger.info(metrics)

    # compare with zero slippage
    logger.info("=" * 60)
    logger.info("COMPARISON: Same strategy with zero execution costs")
    logger.info("=" * 60)

    # reset and run with zero costs
    backtest.reset()
    backtest.executor = ExecutionSimulator(
        config=ExecutionConfig(
            commission_per_share=0.0,
            commission_minimum=0.0,
        ),
        slippage_model=ZeroSlippage(),
    )

    metrics_zero_cost = backtest.run()

    logger.info("Total return (with costs):    %+.2f%%", metrics.total_return * 100)
    logger.info("Total return (zero costs):    %+.2f%%", metrics_zero_cost.total_return * 100)
    cost_impact = metrics_zero_cost.total_return - metrics.total_return
    logger.info("Cost impact:                  %+.2f%%", cost_impact * 100)


def main() -> int:
    """
    Main entry point.
    """
    logger.info("Starting MA crossover backtest")
    try:
        run_backtest()
        return 0
    except Exception as e:
        logger.error("Error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
