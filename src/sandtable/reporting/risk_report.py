"""
src/sandtable/reporting/risk_report.py

Standalone PDF risk report generation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sandtable.config import settings
from sandtable.core.events import RiskAction, RiskBreachEvent
from sandtable.core.result import BacktestResult
from sandtable.reporting import charts


def generate_risk_report(
    result: BacktestResult,
    breach_log: list[RiskBreachEvent] | None = None,
    output_path: str | None = None,
) -> str:
    """
    Generate a standalone PDF risk report.

    Args:
        result: BacktestResult to report on
        breach_log: Optional list of RiskBreachEvent objects
        output_path: Output file path. Defaults to output/risk_report.pdf

    Returns:
        Path to the generated PDF file
    """
    if output_path is None:
        out_dir = settings.BACKTESTER_OUTPUT_DIR
        os.makedirs(out_dir, exist_ok=True)
        output_path = str(out_dir / "risk_report.pdf")
    else:
        os.makedirs(Path(output_path).parent, exist_ok=True)

    eq_df = result.equity_dataframe()

    with PdfPages(output_path) as pdf:
        # drawdown chart
        if not eq_df.empty and len(eq_df) > 1:
            equity_series = eq_df.set_index("timestamp")["equity"]
            running_max = equity_series.cummax()
            drawdown_series = (equity_series - running_max) / running_max

            fig = charts.plot_equity_curve(equity_series, drawdown_series)
            fig.suptitle("Risk Report - Equity & Drawdown", fontsize=14, fontweight="bold")
            pdf.savefig(fig)
            plt.close(fig)

        # breach log table
        breaches = breach_log or result.parameters.get("risk_breaches", [])
        if breaches:
            _page_breach_log(pdf, breaches)

        # summary
        _page_risk_summary(pdf, result, breaches)

    return output_path


def _page_breach_log(pdf: PdfPages, breaches: list[RiskBreachEvent | dict[str, Any]]) -> None:
    """Render risk breach log as a table."""
    fig, ax = plt.subplots(figsize=(12, max(3, min(10, len(breaches) * 0.3 + 2))))
    ax.axis("off")
    ax.set_title("Risk Breach Log", fontsize=14, fontweight="bold", pad=20)

    # limit display to first 30 breaches
    display_breaches = breaches[:30]
    rows = []
    for b in display_breaches:
        if isinstance(b, RiskBreachEvent):
            rows.append([
                str(b.timestamp)[:19],
                b.rule_name,
                b.symbol,
                b.action,
                f"{b.breach_value:.4f}",
                f"{b.threshold:.4f}",
            ])
        elif isinstance(b, dict):
            rows.append([
                str(b.get("timestamp", ""))[:19],
                b.get("rule_name", ""),
                b.get("symbol", ""),
                b.get("action", ""),
                f"{b.get('breach_value', 0):.4f}",
                f"{b.get('threshold', 0):.4f}",
            ])

    if rows:
        table = ax.table(
            cellText=rows,
            colLabels=["Timestamp", "Rule", "Symbol", "Action", "Value", "Threshold"],
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.3)

    if len(breaches) > 30:
        ax.text(0.5, -0.05, f"... and {len(breaches) - 30} more breaches", ha="center", transform=ax.transAxes)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_risk_summary(pdf: PdfPages, result: BacktestResult, breaches: list[RiskBreachEvent | dict[str, Any]]) -> None:
    """Summary statistics page."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    ax.set_title("Risk Summary", fontsize=14, fontweight="bold", pad=20)

    n_rejected = sum(
        1 for b in breaches
        if (isinstance(b, RiskBreachEvent) and b.action == RiskAction.REJECTED)
        or (isinstance(b, dict) and b.get("action") == RiskAction.REJECTED)
    )
    n_resized = sum(
        1 for b in breaches
        if (isinstance(b, RiskBreachEvent) and b.action == RiskAction.RESIZED)
        or (isinstance(b, dict) and b.get("action") == RiskAction.RESIZED)
    )

    rows = [
        ["Max Drawdown", f"{result.metrics.max_drawdown:.2%}"],
        ["Total Risk Breaches", f"{len(breaches)}"],
        ["Orders Rejected", f"{n_rejected}"],
        ["Orders Resized", f"{n_resized}"],
        ["Total Trades", f"{result.metrics.num_trades}"],
    ]

    table = ax.table(
        cellText=rows,
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
