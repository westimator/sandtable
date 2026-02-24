"""
src/sandtable/reporting/comparison.py

PDF strategy comparison report generation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from sandtable.config import settings
from sandtable.reporting import charts

if TYPE_CHECKING:
    from sandtable.core.result import BacktestResult

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf"]


def generate_comparison_report(
    results: dict[str, BacktestResult],
    correlation_matrix: pd.DataFrame | None = None,
    output_path: str | None = None,
) -> str:
    """
    Generate a PDF comparison report for multiple strategies.

    Args:
        results: Dict mapping strategy name to BacktestResult
        correlation_matrix: Optional pre-computed correlation matrix
        output_path: Output file path. Defaults to output/strategy_comparison.pdf

    Returns:
        Path to the generated PDF file
    """
    if output_path is None:
        out_dir = settings.BACKTESTER_OUTPUT_DIR
        os.makedirs(out_dir, exist_ok=True)
        output_path = str(out_dir / "strategy_comparison.pdf")
    else:
        os.makedirs(Path(output_path).parent, exist_ok=True)

    with PdfPages(output_path) as pdf:
        # page 1: overlaid equity curves
        _page_equity_overlay(pdf, results)

        # page 2: metrics comparison table
        _page_metrics_table(pdf, results)

        # page 3: correlation matrix (if provided or computable)
        if correlation_matrix is not None:
            fig = charts.plot_correlation_heatmap(correlation_matrix)
            pdf.savefig(fig)
            plt.close(fig)

    return output_path


def _page_equity_overlay(pdf: PdfPages, results: dict[str, BacktestResult]) -> None:
    """Overlaid equity curves for all strategies."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (name, result) in enumerate(results.items()):
        eq_df = result.equity_dataframe()
        if eq_df.empty:
            continue
        color = COLORS[i % len(COLORS)]
        ax.plot(eq_df["timestamp"], eq_df["equity"], color=color, linewidth=1.2, label=name)

    ax.set_title("Strategy Comparison - Equity Curves")
    ax.set_ylabel("Equity ($)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_metrics_table(pdf: PdfPages, results: dict[str, BacktestResult]) -> None:
    """Side-by-side metrics comparison table."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    ax.set_title("Strategy Metrics Comparison", fontsize=14, fontweight="bold", pad=20)

    metrics_names = [
        "Total Return", "CAGR", "Sharpe Ratio", "Sortino Ratio",
        "Max Drawdown", "Win Rate", "Profit Factor", "Trades",
    ]

    col_labels = ["Metric"] + list(results.keys())
    rows = []
    for metric_name in metrics_names:
        row = [metric_name]
        for result in results.values():
            m = result.metrics
            if metric_name == "Total Return":
                row.append(f"{m.total_return:+.2%}")
            elif metric_name == "CAGR":
                row.append(f"{m.cagr:+.2%}")
            elif metric_name == "Sharpe Ratio":
                row.append(f"{m.sharpe_ratio:.2f}")
            elif metric_name == "Sortino Ratio":
                row.append(f"{m.sortino_ratio:.2f}")
            elif metric_name == "Max Drawdown":
                row.append(f"{m.max_drawdown:.2%}")
            elif metric_name == "Win Rate":
                row.append(f"{m.win_rate:.1%}")
            elif metric_name == "Profit Factor":
                row.append(f"{m.profit_factor:.2f}")
            elif metric_name == "Trades":
                row.append(f"{m.num_trades}")
        rows.append(row)

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
