"""
demo.py

Sandtable v3 showcase script.

Exercises every major feature with zero arguments:
  1. Data loading (CSV fixtures, works offline)
  2. Strategy backtests with realistic execution
  3. Risk management
  4. Parameter sweep
  5. Walk-forward analysis
  6. Statistical significance tests
  7. Strategy comparison
  8. Persistence to SQLite
  9. PDF report generation

Run:
    python demo.py

Reports are saved to output/. Results are persisted to sandtable.db.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# suppress library logging so rich output is clean
os.environ["BACKTESTER_LOG_LEVEL"] = "WARNING"

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import sandtable as st
from sandtable.execution.impact import SquareRootImpactModel
from sandtable.execution.simulator import ExecutionConfig

console = Console()
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

SYMBOLS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT"]
START = "2018-01-01"
END = "2023-12-31"


def step(number: int, total: int, text: str) -> None:
    """Print a step header."""
    console.print(f"\n[bold cyan][{number}/{total}][/bold cyan] {text}")


def metrics_table(result: st.BacktestResult, title: str = "Metrics") -> None:
    """Print a rich table of key metrics."""
    m = result.metrics
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")
    table.add_row("Sharpe Ratio", f"{m.sharpe_ratio:.2f}")
    table.add_row("Sortino Ratio", f"{m.sortino_ratio:.2f}")
    table.add_row("CAGR", f"{m.cagr:.2%}")
    table.add_row("Max Drawdown", f"{m.max_drawdown:.2%}")
    table.add_row("Total Return", f"{m.total_return:.2%}")
    table.add_row("Total Trades", str(m.num_trades))
    table.add_row("Win Rate", f"{m.win_rate:.2%}")
    table.add_row("Profit Factor", f"{m.profit_factor:.2f}")
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sandtable v3 demo")
    parser.add_argument(
        "--store",
        choices=[b.value for b in st.ResultBackend],
        default=st.ResultBackend.SQLITE.value,
        help="Result store backend",
    )
    args = parser.parse_args()

    store_backend = st.ResultBackend(args.store)

    t0 = time.time()

    console.print(Panel("[bold]Sandtable v3 Demo[/bold]", style="bold blue"))

    # persistence
    if store_backend == st.ResultBackend.MYSQL:
        store = st.MySQLResultStore()
        console.print("  [dim]Using MySQL result store[/dim]")
    elif store_backend == st.ResultBackend.SQLITE:
        store = st.SQLiteResultStore(db_path="sandtable.db")
    else:
        raise ValueError(f"Unknown result backend: {store_backend}")

    # execution model
    exec_config = ExecutionConfig(
        commission_per_share=0.005,
        commission_minimum=1.0,
    )
    slippage = st.FixedSlippage(bps=5.0)
    impact = SquareRootImpactModel(eta=0.1)

    # risk rules
    risk_manager = st.RiskManager(rules=[
        st.MaxLeverageRule(max_leverage=2.0),
        st.MaxDrawdownRule(max_drawdown_pct=0.15),
        st.MaxDailyLossRule(max_daily_loss_pct=0.03),
        st.MaxPositionSizeRule(max_position_pct=0.25),
    ])

    ## 1. Load data
    step(1, 9, "Loading market data...")
    provider = st.CSVProvider(data_dir="data/fixtures")
    data = st.DataHandler(
        provider=provider,
        universe=SYMBOLS,
    )
    data.load(START, END)
    console.print(f"  [green]Loaded {', '.join(SYMBOLS)} ({START} to {END})[/green]")

    ## 2. Mean Reversion strategy
    step(2, 9, "Running Mean Reversion strategy...")
    mr_strategy = st.MeanReversionStrategy(lookback=20, threshold=2.0)
    data.reset()
    mr_result = st.run_backtest(
        strategy=mr_strategy,
        data=data,
        initial_capital=100_000,
        position_size_pct=0.10,
        slippage=slippage,
        impact=impact,
        commission=exec_config,
        risk_manager=risk_manager,
        result_store=store,
        result_tags={"demo": "true", "strategy": "mean_reversion"},
    )
    metrics_table(mr_result, "Mean Reversion")

    ## 3. MA Crossover strategy
    step(3, 9, "Running MA Crossover strategy...")
    mac_strategy = st.MACrossoverStrategy(fast_period=10, slow_period=30)
    data.reset()
    mac_result = st.run_backtest(
        strategy=mac_strategy,
        data=data,
        initial_capital=100_000,
        position_size_pct=0.10,
        slippage=slippage,
        impact=impact,
        commission=exec_config,
        risk_manager=risk_manager,
        result_store=store,
        result_tags={"demo": "true", "strategy": "ma_crossover"},
    )
    metrics_table(mac_result, "MA Crossover")

    ## 4. Parameter sweep
    step(4, 9, "Parameter sweep (12 combinations)...")
    data.reset()
    sweep = st.run_parameter_sweep(
        strategy_class=st.MeanReversionStrategy,
        param_grid={
            "lookback": [10, 20, 30, 40],
            "threshold": [1.5, 2.0, 2.5],
        },
        data=data,
        metric=st.Metric.SHARPE_RATIO,
        initial_capital=100_000,
        position_size_pct=0.10,
        slippage=slippage,
        impact=impact,
        commission=exec_config,
        result_store=store,
    )
    console.print(f"  Best: {sweep.best_params} (Sharpe={sweep.best_result.metrics.sharpe_ratio:.2f})")

    sweep_df = sweep.to_dataframe()
    sweep_table = Table(title="Sweep Results (top 5 by Sharpe)", show_header=True, header_style="bold")
    sweep_table.add_column("lookback", justify="right")
    sweep_table.add_column("threshold", justify="right")
    sweep_table.add_column("Sharpe", justify="right")
    sweep_table.add_column("Return", justify="right")
    sweep_table.add_column("Max DD", justify="right")
    top5 = sweep_df.sort_values("sharpe_ratio", ascending=False).head(5)
    for _, row in top5.iterrows():
        sweep_table.add_row(
            str(int(row["lookback"])),
            f"{row['threshold']:.1f}",
            f"{row['sharpe_ratio']:.2f}",
            f"{row['total_return']:.2%}",
            f"{row['max_drawdown']:.2%}",
        )
    console.print(sweep_table)

    ## 5. Walk-forward analysis
    step(5, 9, "Walk-forward analysis...")
    data.reset()
    wf = st.run_walkforward(
        strategy_cls=st.MACrossoverStrategy,
        param_grid={
            "fast_period": [5, 10, 15],
            "slow_period": [20, 30, 40],
        },
        data=data,
        train_window=252,
        test_window=126,
        optimization_metric=st.Metric.SHARPE_RATIO,
        initial_capital=100_000,
        position_size_pct=0.10,
        slippage=slippage,
        impact=impact,
        commission=exec_config,
    )
    wf_table = Table(title="Walk-Forward Folds", show_header=True, header_style="bold")
    wf_table.add_column("Fold")
    wf_table.add_column("Test Period")
    wf_table.add_column("Chosen Params")
    wf_table.add_column("IS Sharpe", justify="right")
    wf_table.add_column("OOS Sharpe", justify="right")
    for fold in wf.folds:
        wf_table.add_row(
            str(fold.fold_index),
            f"{fold.test_start} - {fold.test_end}",
            str(fold.chosen_params),
            f"{fold.in_sample_metrics.get('sharpe_ratio', 0):.2f}",
            f"{fold.out_of_sample_metrics.get('sharpe_ratio', 0):.2f}",
        )
    console.print(wf_table)
    console.print(f"  Aggregate OOS: Sharpe={wf.oos_sharpe:.2f}, CAGR={wf.oos_cagr:.2%}, MaxDD={wf.oos_max_drawdown:.2%}")

    ## 6. Statistical significance tests
    step(6, 9, "Statistical significance tests...")
    for name, result in [("Mean Reversion", mr_result), ("MA Crossover", mac_result)]:
        sig = result.significance_tests(
            n_simulations=500,
            random_seed=42,
        )
        sig_table = Table(
            title=f"Significance - {name}",
            show_header=True,
            header_style="bold",
        )
        sig_table.add_column("Test")
        sig_table.add_column("Statistic", justify="right")
        sig_table.add_column("p-value", justify="right")
        sig_table.add_column("Significant?", justify="center")
        for test_name, sr in sig.items():
            sig_table.add_row(
                test_name,
                f"{sr.observed_statistic:.4f}",
                f"{sr.p_value:.4f}",
                "[green]Yes[/green]" if sr.is_significant else "[red]No[/red]",
            )
        console.print(sig_table)

    ## 7. Strategy comparison
    step(7, 9, "Comparing strategies...")
    comparison = st.run_comparison({
        "Mean Reversion": mr_result,
        "MA Crossover": mac_result,
    })
    console.print(comparison.performance_table.to_string())
    console.print(f"\n  Correlation: {comparison.correlation_matrix.iloc[0, 1]:.2f}")
    if comparison.blended_equity_curve:
        final_blend = comparison.blended_equity_curve[-1].equity
        initial_blend = comparison.blended_equity_curve[0].equity
        blend_return = (final_blend - initial_blend) / initial_blend
        console.print(f"  Blended portfolio return: {blend_return:.2%}")

    ## 8-9. Report generation + persistence summary
    step(8, 9, "Generating reports...")
    reports = []

    path = st.generate_pdf_tearsheet(mr_result, output_path=str(OUTPUT_DIR / "mean_reversion_tearsheet.pdf"))
    reports.append(path)

    path = st.generate_pdf_tearsheet(mac_result, output_path=str(OUTPUT_DIR / "ma_crossover_tearsheet.pdf"))
    reports.append(path)

    path = st.generate_risk_report(mr_result, output_path=str(OUTPUT_DIR / "risk_report.pdf"))
    reports.append(path)

    path = st.generate_comparison_report(
        {"Mean Reversion": mr_result, "MA Crossover": mac_result},
        correlation_matrix=comparison.correlation_matrix,
        output_path=str(OUTPUT_DIR / "strategy_comparison.pdf"),
    )
    reports.append(path)

    for rpath in reports:
        console.print(f"  [green]Created {rpath}[/green]")

    # summary
    step(9, 9, "Persistence summary...")
    runs = store.list_runs(limit=100)
    console.print(f"  [green]{len(runs)} runs persisted to sandtable.db[/green]")

    elapsed = time.time() - t0
    console.print(
        Panel(
            f"Reports saved to [bold]{OUTPUT_DIR}/[/bold]. "
            f"Results persisted to [bold]sandtable.db[/bold]. "
            f"Total runtime: [bold]{elapsed:.1f}s[/bold]",
            style="bold green",
        )
    )


if __name__ == "__main__":
    main()
