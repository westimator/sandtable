"""
sandtable.report - Report generation (tearsheets, comparisons).
"""

from sandtable.report.comparison import compare_strategies
from sandtable.report.tearsheet import generate_tearsheet

__all__ = ["generate_tearsheet", "compare_strategies"]
