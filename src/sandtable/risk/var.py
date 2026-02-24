"""
src/sandtable/risk/var.py

Historical Value-at-Risk computation from daily returns.
"""

from __future__ import annotations


def compute_var(
    daily_returns: list[float],
    lookback: int = 252,
    confidence: float = 0.95,
) -> float:
    """
    Historical VaR using sorted daily returns.

    Returns a negative number representing the worst expected daily loss
    at the given confidence level over the lookback window.

    Args:
        daily_returns: list of daily percentage returns (e.g. 0.01 = 1%)
        lookback: number of most recent returns to use (default 252 = 1 year)
        confidence: confidence level (default 0.95 = 95%)

    Returns:
        VaR as a negative float (e.g. -0.02 means 2% worst daily loss).
        Returns 0.0 if insufficient data.
    """
    if not daily_returns:
        return 0.0

    # use only the most recent lookback returns
    window = daily_returns[-lookback:]

    if len(window) < 2:
        return 0.0

    sorted_returns = sorted(window)

    # percentile index: for 95% confidence, we want the 5th percentile
    index = int((1.0 - confidence) * len(sorted_returns))
    index = max(
        0,
        min(index, len(sorted_returns) - 1)
    )

    return sorted_returns[index]
