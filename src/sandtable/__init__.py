"""
sandtable - Event-driven backtesting framework with realistic execution modeling.
"""

# High-level API
from sandtable.api import SweepResult, run_backtest, run_parameter_sweep

# Configuration & logging
from sandtable.config import BacktestConfig, settings

# Events (needed to write generate_signal)
from sandtable.core.events import Direction, MarketDataEvent, SignalEvent

# Results
from sandtable.core.result import BacktestResult

# Instrument abstraction
from sandtable.data import Currency, Equity, Future, Instrument, InstrumentType, TradingHours, Universe

# Data engine
from sandtable.data_engine import (
    AbstractDataProvider,
    CachingProvider,
    CSVProvider,
    DataHandler,
    YFinanceProvider,
)

# Shared type definitions
from sandtable.data_types import DataSource, Metric, ResultBackend

# Execution
from sandtable.execution.slippage import FixedSlippage

# Metrics & Reports
# Persistence
from sandtable.persistence import (
    AbstractResultStore,
    AbstractSQLResultStore,
    MySQLResultStore,
    RunSummary,
    SQLiteResultStore,
    SyncResult,
    sync_stores,
)
from sandtable.report import compare_strategies
from sandtable.reporting import (
    TCAReport,
    compute_tca,
    generate_comparison_report,
    generate_pdf_tearsheet,
    generate_risk_report,
)

# Research workflow
from sandtable.research import (
    ComparisonResult,
    WalkForwardFold,
    WalkForwardResult,
    run_comparison,
    run_walkforward,
)

# Risk management
from sandtable.risk import (
    AbstractRiskManager,
    MaxConcentrationRule,
    MaxDailyLossRule,
    MaxDrawdownRule,
    MaxLeverageRule,
    MaxOrderSizeRule,
    MaxPortfolioExposureRule,
    MaxPositionSizeRule,
    RiskManager,
    compute_var,
)

# Statistical significance testing
from sandtable.stats import SignificanceResult, run_significance_tests

# Strategy
from sandtable.strategy.abstract_strategy import AbstractStrategy
from sandtable.strategy.buy_and_hold_strategy import BuyAndHoldStrategy
from sandtable.strategy.ma_crossover_strategy import MACrossoverStrategy
from sandtable.strategy.mean_reversion_strategy import MeanReversionStrategy
from sandtable.utils.logger import get_logger

__all__ = [
    # High-level API
    "run_backtest",
    "run_parameter_sweep",
    "BacktestResult",
    "SweepResult",
    # Strategy
    "AbstractStrategy",
    "BuyAndHoldStrategy",
    "MACrossoverStrategy",
    "MeanReversionStrategy",
    # Events
    "Direction",
    "SignalEvent",
    "MarketDataEvent",
    # Instrument abstraction
    "Currency",
    "Equity",
    "Future",
    "Instrument",
    "InstrumentType",
    "TradingHours",
    "Universe",
    # Data engine
    "AbstractDataProvider",
    "CachingProvider",
    "CSVProvider",
    "DataHandler",
    "YFinanceProvider",
    # Execution
    "FixedSlippage",
    # Risk management
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
    # Research workflow
    "BacktestConfig",
    "ComparisonResult",
    "WalkForwardFold",
    "WalkForwardResult",
    "run_comparison",
    "run_walkforward",
    # Statistical significance testing
    "SignificanceResult",
    "run_significance_tests",
    # Metrics & Reports
    "Metric",
    "compare_strategies",
    # PDF Reporting & TCA
    "TCAReport",
    "compute_tca",
    "generate_comparison_report",
    "generate_pdf_tearsheet",
    "generate_risk_report",
    # Persistence
    "AbstractResultStore",
    "AbstractSQLResultStore",
    "MySQLResultStore",
    "RunSummary",
    "SQLiteResultStore",
    "SyncResult",
    "sync_stores",
    # Shared type definitions
    "DataSource",
    "ResultBackend",
    # Configuration & logging
    "settings",
    "get_logger",
]
