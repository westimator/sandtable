"""
sandtable - Event-driven backtesting framework with realistic execution modeling.
"""

# High-level API
from sandtable.api import SweepResult, run_backtest, run_parameter_sweep

# Results
from sandtable.core.result import BacktestResult

# Strategy
from sandtable.strategy.abstract_strategy import AbstractStrategy
from sandtable.strategy.ma_crossover import MACrossoverStrategy
from sandtable.strategy.mean_reversion import MeanReversionStrategy

# Events (needed to write generate_signal)
from sandtable.core.events import Direction, MarketDataEvent, SignalEvent

# Data handlers
from sandtable.data_handlers.csv_data_handler import CSVDataHandler
from sandtable.data_handlers.multi_handler import MultiDataHandler
from sandtable.data_handlers.yfinance_handler import YFinanceDataHandler

# Execution
from sandtable.execution.slippage import FixedSlippage

# Metrics & Reports
from sandtable.metrics import Metric
from sandtable.report import compare_strategies

# Configuration & logging
from sandtable.config import settings
from sandtable.utils.logger import get_logger

__all__ = [
    # High-level API
    "run_backtest",
    "run_parameter_sweep",
    "BacktestResult",
    "SweepResult",
    # Strategy
    "AbstractStrategy",
    "MACrossoverStrategy",
    "MeanReversionStrategy",
    # Events
    "Direction",
    "SignalEvent",
    "MarketDataEvent",
    # Data handlers
    "CSVDataHandler",
    "MultiDataHandler",
    "YFinanceDataHandler",
    # Execution
    "FixedSlippage",
    # Metrics & Reports
    "Metric",
    "compare_strategies",
    # Configuration & logging
    "settings",
    "get_logger",
]
