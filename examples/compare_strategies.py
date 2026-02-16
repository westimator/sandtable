"""
examples/compare_strategies.py

Example: Compare multiple strategy configurations side by side.

Runs the same MA crossover strategy with different parameter sets,
then generates an HTML report with overlaid equity curves and a
metrics comparison table.

Usage:
    uv run python examples/compare_strategies.py
"""

from pathlib import Path

from tabulate import tabulate

from sandtable import (
    CSVDataHandler,
    FixedSlippage,
    MACrossoverStrategy,
    compare_strategies,
    get_logger,
    run_backtest,
    settings,
)

logger = get_logger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CONFIGS = {
    "Fast (5-day / 15-day MA)": {"fast_period": 5, "slow_period": 15},
    "Medium (10-day / 30-day MA)": {"fast_period": 10, "slow_period": 30},
    "Slow (20-day / 50-day MA)": {"fast_period": 20, "slow_period": 50},
}


def main() -> None:
    results = {}

    for label, params in CONFIGS.items():
        logger.info("Running %s ...", label)

        data = CSVDataHandler(DATA_DIR / "sample_ohlcv.csv", "SPY")
        result = run_backtest(
            strategy=MACrossoverStrategy(**params),
            data=data,
            initial_capital=100_000,
            slippage=FixedSlippage(bps=5),
            commission=0.005,
        )
        results[label] = result

    ## Summary table
    rows = []
    for label, r in results.items():
        m = r.metrics
        rows.append({
            "strategy": label,
            "return": f"{m.total_return:+.2%}",
            "sharpe": f"{m.sharpe_ratio:.2f}",
            "sortino": f"{m.sortino_ratio:.2f}",
            "max_dd": f"{m.max_drawdown:.2%}",
            "trades": m.num_trades,
            "win_rate": f"{m.win_rate:.1%}",
        })
    bold_headers = [f"\033[1m{h}\033[0m" for h in rows[0].keys()]
    values = [list(r.values()) for r in rows]
    logger.info(
        "Results:\n%s",
        tabulate(values, headers=bold_headers, tablefmt="rounded_grid"),
    )

    ## Generate comparison report
    output_dir = settings.BACKTESTER_OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)
    out = output_dir / "strategy_comparison.pdf"
    compare_strategies(results, output_path=str(out))
    logger.info("Comparison report saved to %s", out.resolve())


if __name__ == "__main__":
    main()
