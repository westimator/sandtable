"""
src/sandtable/research/

Research workflow: walk-forward analysis, strategy comparison.
"""

from sandtable.research.compare import ComparisonResult, run_comparison
from sandtable.research.walkforward import WalkForwardFold, WalkForwardResult, run_walkforward

__all__ = [
    "ComparisonResult",
    "WalkForwardFold",
    "WalkForwardResult",
    "run_comparison",
    "run_walkforward",
]
