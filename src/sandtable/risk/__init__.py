"""
sandtable.risk - Risk management layer for order filtering and position sizing.

Sits between SIGNAL and ORDER in the event chain. The risk manager can
approve, resize, or reject proposed orders before they reach execution.
"""

from sandtable.risk.abstract_risk_manager import AbstractRiskManager
from sandtable.risk.risk_manager import RiskManager
from sandtable.risk.rules import (
    AbstractRule,
    MaxConcentrationRule,
    MaxDailyLossRule,
    MaxDrawdownRule,
    MaxLeverageRule,
    MaxOrderSizeRule,
    MaxPortfolioExposureRule,
    MaxPositionSizeRule,
)
from sandtable.risk.var import compute_var

__all__ = [
    "AbstractRule",
    "AbstractRiskManager",
    "RiskManager",
    "MaxConcentrationRule",
    "MaxDailyLossRule",
    "MaxDrawdownRule",
    "MaxLeverageRule",
    "MaxOrderSizeRule",
    "MaxPortfolioExposureRule",
    "MaxPositionSizeRule",
    "compute_var",
]
