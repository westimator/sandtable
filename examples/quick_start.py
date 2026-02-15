"""
examples/quick_start.py

Quick start example using the clean research API.

Run with:
    uv run python examples/quick_start.py
"""

from pathlib import Path

from tabulate import tabulate

from sandtable import (
    CSVDataHandler,
    FixedSlippage,
    MeanReversionStrategy,
    Metric,
    get_logger,
    run_backtest,
    run_parameter_sweep,
)

logger = get_logger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    ## 1. Run a single backtest
    logger.info("=" * 60)
    logger.info("SINGLE BACKTEST")
    logger.info("=" * 60)

    data = CSVDataHandler(DATA_DIR / "sample_ohlcv.csv", "SPY")
    result = run_backtest(
        strategy=MeanReversionStrategy(lookback=20, threshold=2.0),
        data=data,
        initial_capital=100_000,
        slippage=FixedSlippage(bps=5),
        commission=0.005,
    )

    logger.info(result.metrics)
    logger.info("Equity curve: %d points", len(result.equity_curve))
    logger.info("Trades: %d", len(result.trades))

    ## 2. Parameter sweep
    logger.info("\n" + "=" * 60)
    logger.info("PARAMETER SWEEP")
    logger.info("=" * 60)

    sweep_data = CSVDataHandler(DATA_DIR / "sample_ohlcv.csv", "SPY")
    sweep = run_parameter_sweep(
        strategy_class=MeanReversionStrategy,
        param_grid={
            "lookback": [10, 20, 30],
            "threshold": [1.5, 2.0, 2.5],
        },
        data=sweep_data,
        metric=Metric.SHARPE_RATIO,
    )

    logger.info("Tested %d combinations", len(sweep.results))
    logger.info("Best params: %s", sweep.best_params)
    logger.info("Best Sharpe: %.4f", sweep.best_result.metrics.sharpe_ratio)
    sweep_df = sweep.to_dataframe()[["lookback", "threshold", Metric.SHARPE_RATIO, Metric.TOTAL_RETURN]]
    bold_headers = [f"\033[1m{c}\033[0m" for c in sweep_df.columns]
    logger.info(
        "Sweep summary:\n%s",
        tabulate(sweep_df.values, headers=bold_headers, tablefmt="rounded_grid", floatfmt=".4f"),
    )


if __name__ == "__main__":
    main()
