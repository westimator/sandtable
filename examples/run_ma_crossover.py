"""
examples/run_ma_crossover.py

Advanced example: manual component wiring for full control.
Most users should use the high-level `run_backtest()` API instead.
See examples/quick_start.py for the recommended approach.

Usage:
    uv run python examples/run_ma_crossover.py
"""

import logging
import sys
from pathlib import Path

from sandtable import CSVDataHandler, MACrossoverStrategy
from sandtable.core import Backtest
from sandtable.execution import ExecutionConfig, ExecutionSimulator, FixedSlippage, ZeroSlippage
from sandtable.portfolio import Portfolio

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> int:
    data_file = DATA_DIR / "sample_ohlcv.csv"
    logger.info("Loading data from: %s", data_file)

    # Set up components manually
    data = CSVDataHandler(data_file, "SPY")
    strategy = MACrossoverStrategy(fast_period=10, slow_period=30)
    portfolio = Portfolio(initial_capital=100_000)
    executor = ExecutionSimulator(
        config=ExecutionConfig(commission_per_share=0.005),
        slippage_model=FixedSlippage(bps=5),
    )

    backtest = Backtest(data, strategy, portfolio, executor)

    logger.info("Running backtest on %d bars of %s...", len(data), data.symbol)
    logger.info("Strategy: MA Crossover (fast=%d, slow=%d)", strategy.fast_period, strategy.slow_period)
    logger.info("Initial capital: $%,.2f", portfolio.initial_capital)

    metrics = backtest.run()

    logger.info("=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)
    logger.info(metrics)

    # Compare with zero slippage
    logger.info("=" * 60)
    logger.info("COMPARISON: Same strategy with zero execution costs")
    logger.info("=" * 60)

    backtest.reset()
    backtest.executor = ExecutionSimulator(
        config=ExecutionConfig(commission_per_share=0.0, commission_minimum=0.0),
        slippage_model=ZeroSlippage(),
    )

    metrics_zero_cost = backtest.run()

    logger.info("Total return (with costs):    %+.2f%%", metrics.total_return * 100)
    logger.info("Total return (zero costs):    %+.2f%%", metrics_zero_cost.total_return * 100)
    cost_impact = metrics_zero_cost.total_return - metrics.total_return
    logger.info("Cost impact:                  %+.2f%%", cost_impact * 100)

    return 0


if __name__ == "__main__":
    sys.exit(main())
