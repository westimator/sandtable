"""
src/sandtable/config.py

Centralised configuration loaded from environment variables,
plus the BacktestConfig dataclass for reproducible run specification.

Create a ``.env`` file in the project root (see ``.env.example``)
or export variables in your shell.  Values that are not set fall
back to sensible defaults.
"""

from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from sandtable.data_types import DataSource
from sandtable.utils.exceptions import ConfigurationError

load_dotenv()

_raw_cache = os.environ.get("BACKTESTER_CACHE_DIR", "")
_raw_output = os.environ.get("BACKTESTER_OUTPUT_DIR", "output")


@dataclass(frozen=True)
class Settings:
    """Framework-wide settings, each backed by an environment variable.

    Construct with no arguments for env-var defaults, or pass explicit
    values to override (useful in tests).
    """

    ## Logging

    BACKTESTER_LOG_LEVEL: str = os.environ.get("BACKTESTER_LOG_LEVEL", "INFO").upper()
    """Python logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    Default: ``INFO``"""

    BACKTESTER_LOG_FORMAT: str = os.environ.get("BACKTESTER_LOG_FORMAT", "[%(asctime)s] %(levelname)s:%(name)s: %(message)s")
    """Format string for log messages (stdlib ``logging`` syntax).
    Default: ``[%(asctime)s] %(levelname)s:%(name)s: %(message)s``"""

    BACKTESTER_LOG_DATE_FORMAT: str = os.environ.get("BACKTESTER_LOG_DATE_FORMAT", "%Y-%m-%d %H:%M:%S")
    """Date format for log timestamps (``strftime`` syntax).
    Default: ``%Y-%m-%d %H:%M:%S``"""

    ## Metrics

    BACKTESTER_RISK_FREE_RATE: float = float(os.environ.get("BACKTESTER_RISK_FREE_RATE", "0.0"))
    """Annual risk-free rate for Sharpe / Sortino calculation.
    Default: ``0.0``"""

    BACKTESTER_TRADING_DAYS: int = int(os.environ.get("BACKTESTER_TRADING_DAYS", "252"))
    """Trading days per year for annualisation (252 equity, 365 crypto).
    Default: ``252``"""

    ## Data

    BACKTESTER_CACHE_DIR: Path | None = Path(_raw_cache) if _raw_cache else None
    """Default directory for caching downloaded data (e.g. yfinance).
    Set to an empty string to disable.
    Default: *unset* (caching disabled)"""

    BACKTESTER_OUTPUT_DIR: Path = Path(_raw_output)
    """Directory for script outputs (tearsheets, charts, etc.).
    Default: ``outputs``"""

    def __post_init__(self) -> None:
        """Validate settings on creation."""
        _VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.BACKTESTER_LOG_LEVEL not in _VALID_LOG_LEVELS:
            raise ConfigurationError(
                f"BACKTESTER_LOG_LEVEL={self.BACKTESTER_LOG_LEVEL!r} — "
                f"must be one of {sorted(_VALID_LOG_LEVELS)}"
            )

        if not (0.0 <= self.BACKTESTER_RISK_FREE_RATE < 1.0):
            raise ConfigurationError(
                f"BACKTESTER_RISK_FREE_RATE={self.BACKTESTER_RISK_FREE_RATE} — "
                f"must be a decimal in [0.0, 1.0) (e.g. 0.05 for 5%)"
            )

        if not (1 <= self.BACKTESTER_TRADING_DAYS <= 365):
            raise ConfigurationError(
                f"BACKTESTER_TRADING_DAYS={self.BACKTESTER_TRADING_DAYS} — "
                f"must be between 1 and 365"
            )

    @property
    def BACKTESTER_LOG_LEVEL_INT(self) -> int:
        """Return the logging level as an ``int`` usable by the stdlib."""
        return getattr(logging, self.BACKTESTER_LOG_LEVEL, logging.INFO)


settings = Settings()


# BacktestConfig, a serializable specification for a single backtest run

def _class_to_str(cls: type) -> str:
    """Serialize a class to 'module.ClassName' string."""
    return f"{cls.__module__}.{cls.__qualname__}"


def _str_to_class(s: str) -> type:
    """Resolve a 'module.ClassName' string back to a class."""
    module_path, _, class_name = s.rpartition(".")
    mod = importlib.import_module(name=module_path)
    return getattr(mod, class_name)


@dataclass(frozen=True)
class BacktestConfig:
    """
    Frozen, serializable specification for a single backtest run.

    Every field needed to reproduce a backtest is captured here.
    Strategy class is stored by reference and serialized as a
    'module.ClassName' string for JSON round-tripping.
    """

    strategy_cls: type
    strategy_params: dict[str, Any] = field(default_factory=dict)
    universe: list[str] = field(default_factory=lambda: ["SPY"])
    start_date: str = "2018-01-01"
    end_date: str = "2023-12-31"
    initial_capital: float = 100_000.0
    position_size_pct: float = 0.10
    data_source: DataSource | str = DataSource.CSV

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "strategy_cls": _class_to_str(self.strategy_cls),
            "strategy_params": dict(self.strategy_params),
            "universe": list(self.universe),
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "position_size_pct": self.position_size_pct,
            "data_source": self.data_source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BacktestConfig:
        """Reconstruct from a dict (as produced by to_dict)."""
        return cls(
            strategy_cls=_str_to_class(data["strategy_cls"]),
            strategy_params=data.get("strategy_params", {}),
            universe=data.get("universe", ["SPY"]),
            start_date=data.get("start_date", "2018-01-01"),
            end_date=data.get("end_date", "2023-12-31"),
            initial_capital=data.get("initial_capital", 100_000.0),
            position_size_pct=data.get("position_size_pct", 0.10),
            data_source=data.get("data_source", "csv"),
        )
