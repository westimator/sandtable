"""
sandtable.report - Report generation (tearsheets, comparisons).
"""

from sandtable.report.tearsheet import generate_tearsheet
from sandtable.report.comparison import compare_strategies

__all__ = ["generate_tearsheet", "compare_strategies"]
