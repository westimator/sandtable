"""
sandtable.reporting - PDF report generation using matplotlib.

Separate from sandtable.report (HTML reports). This module produces
multi-page PDF tearsheets, TCA breakdowns, risk reports, and
strategy comparison documents via matplotlib PdfPages.
"""

from sandtable.reporting.comparison import generate_comparison_report
from sandtable.reporting.risk_report import generate_risk_report
from sandtable.reporting.tca import TCAReport, compute_tca
from sandtable.reporting.tearsheet import generate_pdf_tearsheet

__all__ = [
    "TCAReport",
    "compute_tca",
    "generate_comparison_report",
    "generate_pdf_tearsheet",
    "generate_risk_report",
]
