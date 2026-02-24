"""
sandtable.stats - Statistical significance testing for backtest results.

Answers: "Is this backtest result statistically significant or just noise?"
"""

from sandtable.stats.significance import (
    SignificanceResult,
    bootstrap_sharpe,
    permutation_test_sharpe,
    run_significance_tests,
    ttest_mean_return,
)

__all__ = [
    "SignificanceResult",
    "run_significance_tests",
    "permutation_test_sharpe",
    "bootstrap_sharpe",
    "ttest_mean_return",
]
