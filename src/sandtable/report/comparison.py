"""
src/sandtable/report/comparison.py

Strategy comparison report generator.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from sandtable.data_types.metric import Metric

if TYPE_CHECKING:
    from sandtable.core.result import BacktestResult

COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]


def compare_strategies(
    results: dict[str, BacktestResult],
    output_path: str | None = None,
) -> str:
    """
    Generate an HTML comparison of multiple backtest results.

    Args:
        results: Dict mapping strategy name to BacktestResult
        output_path: If provided, write HTML to this file

    Returns:
        HTML string
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Overlaid equity curves
    for i, (name, result) in enumerate(results.items()):
        eq_df = result.equity_dataframe()
        color = COLORS[i % len(COLORS)]
        ax.plot(eq_df["timestamp"], eq_df["equity"], label=name, color=color, linewidth=2)

    ax.set_title("Equity Curves", fontsize=13, fontweight="bold")
    ax.set_ylabel("Equity ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    # Metrics comparison table
    metric_names: list[Metric | str] = [
        Metric.TOTAL_RETURN, Metric.CAGR, Metric.SHARPE_RATIO, Metric.SORTINO_RATIO,
        Metric.MAX_DRAWDOWN, "num_trades", Metric.WIN_RATE, Metric.PROFIT_FACTOR,
    ]
    strategy_names = list(results.keys())

    SIGNED_PCT_METRICS = {Metric.TOTAL_RETURN, Metric.CAGR}

    header = "<tr><th>Metric</th>" + "".join(f"<th>{n}</th>" for n in strategy_names) + "</tr>"
    rows = []
    for metric in metric_names:
        cells = f"<td>{metric}</td>"
        for name in strategy_names:
            m = results[name].metrics
            val = getattr(m, metric)
            if metric == Metric.MAX_DRAWDOWN:
                cells += f"<td>{-val:.2%}</td>"
            elif metric in SIGNED_PCT_METRICS:
                cells += f"<td>{val:+.2%}</td>"
            elif metric == Metric.WIN_RATE:
                cells += f"<td>{val:.1%}</td>"
            elif metric == "num_trades":
                cells += f"<td>{val}</td>"
            else:
                cells += f"<td>{val:.2f}</td>"
        rows.append(f"<tr>{cells}</tr>")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Strategy Comparison</title>
<style>
  .bt-comparison {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                    max-width: 1100px; margin: 0 auto; padding: 20px; background: #fafafa; }}
  .bt-comparison h2 {{ text-align: center; color: #222; }}
  .bt-comparison img {{ width: 100%; border-radius: 6px; border: 1px solid #e0e0e0; margin-bottom: 20px; }}
  .bt-comparison table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .bt-comparison th {{ background: #37474F; color: white; padding: 8px 12px; text-align: center; }}
  .bt-comparison th:first-child {{ text-align: left; }}
  .bt-comparison td {{ padding: 6px 12px; border-bottom: 1px solid #eee; color: #222; text-align: center; }}
  .bt-comparison td:first-child {{ text-align: left; }}
  .bt-comparison tr:nth-child(even) {{ background: #f5f5f5; }}
</style>
</head>
<body>
<div class="bt-comparison">
<h2>Strategy Comparison</h2>
<img src="data:image/png;base64,{img_b64}" alt="Equity Curves">
<table>
{header}
{"".join(rows)}
</table>
</div>
</body>
</html>
"""

    if output_path:
        path = Path(output_path)
        if path.suffix.lower() == ".pdf":
            _write_pdf(html, path)
        else:
            path.write_text(html)

    return html


def _write_pdf(html: str, path: Path) -> None:
    """Convert HTML comparison report to PDF via weasyprint."""
    from weasyprint import HTML

    HTML(string=html).write_pdf(str(path))
