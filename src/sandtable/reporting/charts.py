"""
src/sandtable/reporting/charts.py

Reusable matplotlib chart generators for PDF reports.

Each function returns a matplotlib Figure object suitable for
embedding in a PdfPages document.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def plot_equity_curve(equity_series: pd.Series, drawdown_series: pd.Series | None = None) -> Figure:
    """
    Equity line with optional drawdown shaded area overlay on secondary y-axis.
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(equity_series.index, equity_series.values, color="#1f77b4", linewidth=1.2, label="Equity")
    ax1.set_ylabel("Equity ($)")
    ax1.set_xlabel("Date")
    ax1.set_title("Equity Curve")
    ax1.grid(True, alpha=0.3)

    if drawdown_series is not None and len(drawdown_series) > 0:
        ax2 = ax1.twinx()
        ax2.fill_between(
            drawdown_series.index,
            drawdown_series.values * 100,
            0,
            alpha=0.3,
            color="red",
            label="Drawdown",
        )
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_ylim(bottom=drawdown_series.min() * 100 * 1.3 if drawdown_series.min() < 0 else -1, top=0)

    fig.tight_layout()
    return fig


def plot_monthly_returns_heatmap(monthly_returns: pd.DataFrame) -> Figure:
    """
    Heatmap of monthly returns. Rows=years, columns=months.

    Args:
        monthly_returns: DataFrame with year index, month columns (1-12), values as decimals.
    """
    fig, ax = plt.subplots(figsize=(10, max(3, len(monthly_returns) * 0.5 + 1)))

    data = monthly_returns.values * 100
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-5, vmax=5)

    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(monthly_returns)))
    ax.set_yticklabels(monthly_returns.index)

    # annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not math.isnan(val):
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=7,
                        color="black" if abs(val) < 3 else "white")

    fig.colorbar(im, ax=ax, label="Return (%)")
    ax.set_title("Monthly Returns")
    fig.tight_layout()
    return fig


def plot_rolling_sharpe(daily_returns: pd.Series, window: int = 252) -> Figure:
    """
    Rolling Sharpe ratio time series.
    """
    if len(daily_returns) < window:
        window = max(20, len(daily_returns) // 2)

    rolling_mean = daily_returns.rolling(window).mean()
    rolling_std = daily_returns.rolling(window).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
    rolling_sharpe = rolling_sharpe.dropna()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, color="#1f77b4", linewidth=1)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title(f"Rolling {window}-Day Sharpe Ratio")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_pnl_distribution(trade_pnls: list[float]) -> Figure:
    """
    Histogram of trade PnLs.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    if trade_pnls:
        ax.hist(trade_pnls, bins=min(50, max(10, len(trade_pnls) // 3)), color="#1f77b4", edgecolor="white", alpha=0.8)
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.7)
    ax.set_title("Trade PnL Distribution")
    ax.set_xlabel("PnL ($)")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_cumulative_pnl(trade_pnls: list[float]) -> Figure:
    """
    Cumulative PnL by trade number.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    if trade_pnls:
        cumulative = np.cumsum(trade_pnls)
        ax.plot(range(1, len(cumulative) + 1), cumulative, color="#1f77b4", linewidth=1.2)
        ax.fill_between(range(1, len(cumulative) + 1), cumulative, 0, alpha=0.1, color="#1f77b4")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Cumulative PnL by Trade")
    ax.set_xlabel("Trade Number")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_cost_decomposition(cost_by_component: dict[str, float]) -> Figure:
    """
    Stacked bar of cost components (slippage, impact, commission).
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    components = list(cost_by_component.keys())
    values = list(cost_by_component.values())
    colors = ["#ff7f0e", "#d62728", "#1f77b4"]

    ax.bar(components, values, color=colors[:len(components)], edgecolor="white")
    ax.set_title("Transaction Cost Decomposition")
    ax.set_ylabel("Cost ($)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_var_timeseries(var_series: pd.Series) -> Figure:
    """
    Daily VaR over time.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(var_series.index, var_series.values * 100, color="#d62728", linewidth=1)
    ax.set_title("Historical Value-at-Risk (95%)")
    ax.set_ylabel("VaR (%)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_leverage_timeseries(leverage_series: pd.Series) -> Figure:
    """
    Leverage over time.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(leverage_series.index, leverage_series.values, color="#ff7f0e", linewidth=1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Portfolio Leverage")
    ax.set_ylabel("Leverage")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_null_distribution(simulated_stats: list[float], observed_stat: float, test_name: str) -> Figure:
    """
    Histogram of simulated null distribution with observed statistic as vertical line.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    if simulated_stats:
        ax.hist(simulated_stats, bins=50, color="#1f77b4", edgecolor="white", alpha=0.7, density=True)
    ax.axvline(x=observed_stat, color="red", linewidth=2, linestyle="--", label=f"Observed: {observed_stat:.3f}")
    ax.set_title(f"Null Distribution - {test_name}")
    ax.set_xlabel("Statistic")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> Figure:
    """
    Strategy correlation matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=(max(5, len(corr_matrix) + 2), max(4, len(corr_matrix) + 1)))
    im = ax.imshow(corr_matrix.values, cmap="RdYlBu_r", vmin=-1, vmax=1, aspect="auto")

    labels = list(corr_matrix.columns)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    # annotate
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr_matrix.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(val) > 0.5 else "black")

    fig.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Return Correlation Matrix")
    fig.tight_layout()
    return fig
