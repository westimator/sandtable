"""
src/sandtable/report/tearsheet.py

HTML tearsheet generator using matplotlib for static charts.
"""

from __future__ import annotations

import base64
import io
import math
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from sandtable.core.result import BacktestResult


def generate_tearsheet(
    result: BacktestResult,
    output_path: str | None = None,
    include_stats: bool = False,
    n_simulations: int = 1_000,
    random_seed: int | None = None,
) -> str:
    """
    Generate a self-contained HTML tearsheet with embedded matplotlib charts.

    Args:
        result: BacktestResult to visualize
        output_path: If provided, write HTML to this file
        include_stats: If True, run significance tests and append results
        n_simulations: Number of simulations for significance tests
        random_seed: Optional seed for significance test reproducibility

    Returns:
        HTML string
    """
    eq_df = result.equity_dataframe()
    trades_df = result.trades_dataframe()
    m = result.metrics

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    start_date = eq_df["timestamp"].iloc[0].strftime("%b %Y")
    end_date = eq_df["timestamp"].iloc[-1].strftime("%b %Y")
    symbols_str = ", ".join(result.symbols)
    chart_title = f"Backtest Tearsheet: {symbols_str} | {start_date} - {end_date} | {len(eq_df)} bars"

    fig.suptitle(chart_title, fontsize=14, fontweight="bold")

    # 1. Equity curve
    ax = axes[0, 0]
    ax.plot(eq_df["timestamp"], eq_df["equity"], color="#2196F3", linewidth=1.5)
    ax.set_title("Equity Curve")
    ax.set_ylabel("Equity ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.grid(True, alpha=0.3)

    # 2. Drawdown
    ax = axes[0, 1]
    peak = eq_df["equity"].cummax()
    drawdown = (eq_df["equity"] - peak) / peak
    ax.fill_between(eq_df["timestamp"], 0, drawdown, color="#F44336", alpha=0.4)
    ax.plot(eq_df["timestamp"], drawdown, color="#D32F2F", linewidth=1)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.grid(True, alpha=0.3)

    # 3. Rolling Sharpe (63-day)
    ax = axes[1, 0]
    if len(eq_df) > 63:
        daily_returns = eq_df["equity"].pct_change().dropna()
        rolling_mean = daily_returns.rolling(63).mean()
        rolling_std = daily_returns.rolling(63).std()
        rolling_sharpe = (rolling_mean / rolling_std) * math.sqrt(252)
        ax.plot(eq_df["timestamp"].iloc[1:], rolling_sharpe, color="#4CAF50", linewidth=1.5)
    ax.axhline(y=0, linestyle="--", color="gray", alpha=0.5)
    ax.set_title("Rolling Sharpe (63-day)")
    ax.set_ylabel("Sharpe Ratio")
    ax.grid(True, alpha=0.3)

    # 4. Daily returns distribution
    ax = axes[1, 1]
    if len(eq_df) > 2:
        daily_returns = eq_df["equity"].pct_change().dropna()
        ax.hist(daily_returns, bins=50, color="#9C27B0", alpha=0.7, edgecolor="white")
    ax.set_title("Daily Returns Distribution")
    ax.set_xlabel("Return")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)

    for a in axes.flat:
        for label in a.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")

    plt.tight_layout()

    # Convert figure to base64 PNG
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    # Build metrics summary
    metrics_text = (
        f"Total Return: {m.total_return:+.2%}  |  "
        f"CAGR: {m.cagr:+.2%}  |  "
        f"Sharpe: {m.sharpe_ratio:.2f}  |  "
        f"Sortino: {m.sortino_ratio:.2f}  |  "
        f"Max DD: {m.max_drawdown:.2%}  |  "
        f"Trades: {m.num_trades}  |  "
        f"Win Rate: {m.win_rate:.1%}  |  "
        f"Profit Factor: {m.profit_factor:.2f}"
    )

    # Build trade log HTML table
    trade_table = ""
    if not trades_df.empty:
        trade_table = "<h3>Trade Log</h3>\n" + trades_df.to_html(
            index=False, classes="trade-table", float_format="%.4f"
        )

    # Build statistical significance table
    stats_table = ""
    if include_stats:
        from sandtable.stats.significance import run_significance_tests

        sig_results = run_significance_tests(
            result,
            n_simulations=n_simulations,
            random_seed=random_seed,
        )
        stats_rows = ""
        for key in ("permutation", "t_test", "bootstrap"):
            sr = sig_results[key]
            sig_text = "Yes" if sr.is_significant else "No"
            z_text = f"{sr.z_score:.2f}" if sr.null_std > 0 else "--"
            stats_rows += (
                f"<tr>"
                f"<td style='text-align:left'>{sr.test_name}</td>"
                f"<td>{sr.observed_statistic:.4f}</td>"
                f"<td>{sr.p_value:.4f}</td>"
                f"<td>{sig_text}</td>"
                f"<td>{z_text}</td>"
                f"</tr>\n"
            )
        stats_table = (
            '<div class="bt-stats">\n'
            "<h3>Statistical Significance</h3>\n"
            '<table class="stats-table">\n'
            "<thead><tr>"
            "<th style='text-align:left'>Test</th>"
            "<th>Statistic</th>"
            "<th>p-value</th>"
            "<th>Significant?</th>"
            "<th>Z-score</th>"
            "</tr></thead>\n"
            f"<tbody>{stats_rows}</tbody>\n"
            "</table>\n"
            "</div>"
        )

    title = f"Backtest Tearsheet: {symbols_str} | {start_date} - {end_date}"
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  .bt-tearsheet {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                   max-width: 1100px; margin: 0 auto; padding: 20px; background: #fafafa; }}
  .bt-tearsheet .metrics {{ background: #fff; padding: 12px 18px; border-radius: 6px;
                            border: 1px solid #e0e0e0; margin-bottom: 20px; font-size: 13px;
                            color: #333; text-align: center; }}
  .bt-tearsheet img {{ width: 100%; border-radius: 6px; border: 1px solid #e0e0e0; }}
  .bt-tearsheet .trade-table {{ width: 100%; border-collapse: collapse; font-size: 11px; margin-top: 8px; }}
  .bt-tearsheet h3 {{ color: #222; }}
  .bt-tearsheet .trade-table th {{ background: #37474F; color: white; padding: 6px 10px; text-align: center; }}
  .bt-tearsheet .trade-table th:first-child {{ text-align: left; }}
  .bt-tearsheet .trade-table td {{ padding: 5px 10px; border-bottom: 1px solid #eee; color: #222; text-align: center; }}
  .bt-tearsheet .trade-table td:first-child {{ text-align: left; }}
  .bt-tearsheet .trade-table tr:nth-child(even) {{ background: #f5f5f5; }}
  .bt-tearsheet .bt-stats {{ margin-top: 20px; }}
  .bt-tearsheet .stats-table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px; }}
  .bt-tearsheet .stats-table th {{ background: #37474F; color: white; padding: 6px 10px; text-align: center; }}
  .bt-tearsheet .stats-table td {{ padding: 5px 10px; border-bottom: 1px solid #eee; color: #222; text-align: center; }}
  .bt-tearsheet .stats-table tr:nth-child(even) {{ background: #f5f5f5; }}
</style>
</head>
<body>
<div class="bt-tearsheet">
<div class="metrics">{metrics_text}</div>
<img src="data:image/png;base64,{img_b64}" alt="Tearsheet Charts">
{trade_table}
{stats_table}
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
    """Convert HTML tearsheet to PDF via weasyprint."""
    from weasyprint import HTML

    HTML(string=html).write_pdf(str(path))
