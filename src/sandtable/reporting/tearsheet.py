"""
src/sandtable/reporting/tearsheet.py

Multi-page PDF tearsheet generation using matplotlib PdfPages.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from sandtable.config import settings
from sandtable.core.events import Direction
from sandtable.core.result import BacktestResult
from sandtable.reporting import charts
from sandtable.reporting.tca import compute_tca


def generate_pdf_tearsheet(
    result: BacktestResult,
    output_path: str | None = None,
    include_stats: bool = False,
    n_simulations: int = 1_000,
    random_seed: int | None = None,
) -> str:
    """
    Generate a multi-page PDF tearsheet for a backtest result.

    Args:
        result: BacktestResult to report on
        output_path: Output file path. Defaults to output/{strategy}_tearsheet.pdf
        include_stats: Whether to include significance test page
        n_simulations: Number of simulations for significance tests
        random_seed: Seed for significance test reproducibility

    Returns:
        Path to the generated PDF file
    """
    if output_path is None:
        strategy_name = result.parameters.get("strategy", "backtest")
        out_dir = settings.BACKTESTER_OUTPUT_DIR
        os.makedirs(out_dir, exist_ok=True)
        output_path = str(out_dir / f"{strategy_name}_tearsheet.pdf")
    else:
        os.makedirs(Path(output_path).parent, exist_ok=True)

    eq_df = result.equity_dataframe()

    with PdfPages(output_path) as pdf:
        # page 1: performance overview
        _page_performance_overview(pdf, result, eq_df)

        # page 2: monthly returns and drawdowns
        _page_monthly_returns(pdf, eq_df)

        # page 3: trade analysis
        _page_trade_analysis(pdf, result)

        # page 4: transaction cost analysis
        _page_tca(pdf, result)

        # page 5: statistical significance (optional)
        if include_stats:
            _page_significance(pdf, result, n_simulations, random_seed)

    return output_path


def _page_performance_overview(pdf: PdfPages, result: BacktestResult, eq_df: pd.DataFrame) -> None:
    """Page 1: equity curve with drawdown + key metrics table."""
    if eq_df.empty:
        return

    equity_series = eq_df.set_index("timestamp")["equity"]

    # compute drawdown series
    running_max = equity_series.cummax()
    drawdown_series = (equity_series - running_max) / running_max

    fig = charts.plot_equity_curve(equity_series, drawdown_series)
    pdf.savefig(fig)
    plt.close(fig)

    # metrics summary page
    m = result.metrics
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    ax.set_title("Performance Summary", fontsize=16, fontweight="bold", pad=20)

    metrics_data = [
        ["Total Return", f"{m.total_return:+.2%}"],
        ["CAGR", f"{m.cagr:+.2%}"],
        ["Sharpe Ratio", f"{m.sharpe_ratio:.2f}"],
        ["Sortino Ratio", f"{m.sortino_ratio:.2f}"],
        ["Max Drawdown", f"{m.max_drawdown:.2%}"],
        ["Win Rate", f"{m.win_rate:.1%}"],
        ["Profit Factor", f"{m.profit_factor:.2f}"],
        ["Total Trades", f"{m.num_trades}"],
        ["Avg Trade PnL", f"${m.avg_trade_pnl:.2f}"],
        ["Trading Days", f"{m.num_days}"],
    ]

    table = ax.table(
        cellText=metrics_data,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_monthly_returns(pdf: PdfPages, eq_df: pd.DataFrame) -> None:
    """Page 2: monthly returns heatmap + rolling Sharpe."""
    if eq_df.empty or len(eq_df) < 2:
        return

    equity_series = eq_df.set_index("timestamp")["equity"]
    daily_returns = equity_series.pct_change().dropna()

    # monthly returns heatmap
    monthly = daily_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    if len(monthly) > 0:
        monthly_df = pd.DataFrame({"return": monthly})
        monthly_df["year"] = monthly_df.index.year
        monthly_df["month"] = monthly_df.index.month
        pivot = monthly_df.pivot_table(index="year", columns="month", values="return", aggfunc="first")

        # ensure all 12 months present
        for m in range(1, 13):
            if m not in pivot.columns:
                pivot[m] = float("nan")
        pivot = pivot[sorted(pivot.columns)]

        fig = charts.plot_monthly_returns_heatmap(pivot)
        pdf.savefig(fig)
        plt.close(fig)

    # rolling Sharpe
    if len(daily_returns) > 20:
        fig = charts.plot_rolling_sharpe(daily_returns)
        pdf.savefig(fig)
        plt.close(fig)


def _page_trade_analysis(pdf: PdfPages, result: BacktestResult) -> None:
    """Page 3: PnL distribution + cumulative PnL."""
    if not result.trades:
        return

    # estimate per-fill PnL from fill price * quantity * direction sign
    # (this is approximate since we don't have round-trip pairing)
    trade_pnls = []
    for t in result.trades:
        sign = 1 if t.direction == Direction.SHORT else -1
        pnl = sign * t.quantity * t.fill_price - t.total_cost
        trade_pnls.append(pnl)

    fig = charts.plot_pnl_distribution(trade_pnls)
    pdf.savefig(fig)
    plt.close(fig)

    fig = charts.plot_cumulative_pnl(trade_pnls)
    pdf.savefig(fig)
    plt.close(fig)


def _page_tca(pdf: PdfPages, result: BacktestResult) -> None:
    """Page 4: transaction cost analysis."""
    if not result.trades:
        return

    gross_pnl = result.metrics.end_equity - result.metrics.start_equity
    tca = compute_tca(result.trades, gross_pnl)

    # cost decomposition bar chart
    if tca.total_cost > 0:
        fig = charts.plot_cost_decomposition(tca.cost_by_component)
        pdf.savefig(fig)
        plt.close(fig)

    # cost summary table
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    ax.set_title("Transaction Cost Summary", fontsize=14, fontweight="bold", pad=20)

    rows = [
        ["Slippage", f"${tca.total_slippage:.2f}", f"${tca.total_slippage / max(len(result.trades), 1):.2f}"],
        ["Market Impact", f"${tca.total_impact:.2f}", f"${tca.total_impact / max(len(result.trades), 1):.2f}"],
        ["Commission", f"${tca.total_commission:.2f}", f"${tca.total_commission / max(len(result.trades), 1):.2f}"],
        ["Total", f"${tca.total_cost:.2f}", f"${tca.cost_per_trade:.2f}"],
    ]

    table = ax.table(
        cellText=rows,
        colLabels=["Component", "Total ($)", "Per Trade ($)"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_significance(pdf: PdfPages, result: BacktestResult, n_simulations: int, random_seed: int | None) -> None:
    """Page 5: statistical significance tests."""
    sig_results = result.significance_tests(
        n_simulations=n_simulations,
        random_seed=random_seed,
    )

    # results table
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    ax.set_title("Statistical Significance Tests", fontsize=14, fontweight="bold", pad=20)

    rows = []
    for key, sr in sig_results.items():
        rows.append([
            sr.test_name,
            f"{sr.observed_statistic:.4f}",
            f"{sr.p_value:.4f}",
            "Yes" if sr.is_significant else "No",
            f"{sr.z_score:.2f}" if sr.z_score != 0 else "--",
        ])

    table = ax.table(
        cellText=rows,
        colLabels=["Test", "Statistic", "p-value", "Significant?", "Z-score"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # null distribution for permutation test
    perm = sig_results.get("permutation")
    if perm and perm.simulated_statistics:
        fig = charts.plot_null_distribution(perm.simulated_statistics, perm.observed_statistic, perm.test_name)
        pdf.savefig(fig)
        plt.close(fig)
