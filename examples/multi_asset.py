"""
examples/multi_asset.py

Example: Run a backtest across multiple tickers using YFinanceDataHandler.

Downloads 2 years of daily data for AAPL, MSFT, and GOOGL, merges
them via MultiDataHandler, and runs an MA crossover strategy on the
combined stream.

Requires internet access (Yahoo Finance).

Usage:
    uv run python examples/multi_asset.py
"""

from tabulate import tabulate

from sandtable import (
    FixedSlippage,
    MACrossoverStrategy,
    MultiDataHandler,
    YFinanceDataHandler,
    get_logger,
    run_backtest,
    settings,
)

logger = get_logger(__name__)

SYMBOLS = ["AAPL", "MSFT", "GOOGL"]
START = "2023-01-01"
END = "2024-12-31"


def main() -> None:
    ## 1. Build per-symbol handlers
    logger.info("Downloading data for %s ...", SYMBOLS)

    handlers = {
        sym: YFinanceDataHandler(symbol=sym, start=START, end=END)
        for sym in SYMBOLS
    }

    ## 2. Merge into a single multi-asset stream
    data = MultiDataHandler(handlers)
    logger.info("Loaded %d total bars across %d symbols", len(data), len(SYMBOLS))

    ## 3. Run backtest
    result = run_backtest(
        strategy=MACrossoverStrategy(fast_period=10, slow_period=30),
        data=data,
        initial_capital=100_000,
        slippage=FixedSlippage(bps=5),
        commission=0.005,
    )

    ## 4. Results
    logger.info(result)

    # Trade log
    trades_df = result.trades_dataframe()
    if not trades_df.empty:
        log = trades_df[["timestamp", "symbol", "direction", "quantity", "fill_price"]].copy()
        log["timestamp"] = log["timestamp"].dt.strftime("%Y-%m-%d")
        log["fill_price"] = log["fill_price"].map("${:,.2f}".format)
        log.columns = ["date", "symbol", "direction", "shares", "price"]
        bold_headers = [f"\033[1m{c}\033[0m" for c in log.columns]
        logger.info(
            "Trade log:\n%s",
            tabulate(log.values, headers=bold_headers, tablefmt="rounded_grid"),
        )

        # Per-symbol fill counts
        counts = trades_df.groupby("symbol").size().reset_index(name="fills")
        bold_headers = [f"\033[1m{c}\033[0m" for c in counts.columns]
        logger.info(
            "Fills per symbol:\n%s",
            tabulate(counts.values, headers=bold_headers, tablefmt="rounded_grid"),
        )

    # Generate tearsheet
    output_dir = settings.BACKTESTER_OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)
    out = output_dir / "multi_asset_tearsheet.html"
    result.tearsheet(output_path=str(out))
    logger.info("Tearsheet saved to %s", out.resolve())


if __name__ == "__main__":
    main()
