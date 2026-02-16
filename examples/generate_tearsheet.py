"""
examples/generate_tearsheet.py

Example: Run a backtest and generate a PDF tearsheet.

The tearsheet includes equity curve, drawdown, rolling Sharpe,
returns distribution, key metrics, and a trade log.

Usage:
    uv run python examples/generate_tearsheet.py
"""

from sandtable import (
    FixedSlippage,
    MACrossoverStrategy,
    YFinanceDataHandler,
    get_logger,
    run_backtest,
    settings,
)

logger = get_logger(__name__)


def main() -> None:
    ## 1. Run backtest
    data = YFinanceDataHandler(symbol="SPY", start="2022-01-01", end="2024-12-31")

    result = run_backtest(
        strategy=MACrossoverStrategy(fast_period=10, slow_period=30),
        data=data,
        initial_capital=100_000,
        slippage=FixedSlippage(bps=5),
        commission=0.005,
    )

    logger.info(result.metrics)

    ## 2. Generate tearsheet
    output_dir = settings.BACKTESTER_OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)
    out = output_dir / "tearsheet.pdf"
    result.tearsheet(output_path=str(out))
    logger.info("Tearsheet written to %s", out.resolve())


if __name__ == "__main__":
    main()
