"""
src/sandtable/config.py

Centralised configuration loaded from environment variables.

Create a ``.env`` file in the project root (see ``.env.example``)
or export variables in your shell.  Values that are not set fall
back to sensible defaults.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_raw_cache = os.environ.get("BACKTESTER_CACHE_DIR", "")
_raw_output = os.environ.get("BACKTESTER_OUTPUT_DIR", "outputs")


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
            raise ValueError(
                f"BACKTESTER_LOG_LEVEL={self.BACKTESTER_LOG_LEVEL!r} — "
                f"must be one of {sorted(_VALID_LOG_LEVELS)}"
            )

        if not (0.0 <= self.BACKTESTER_RISK_FREE_RATE < 1.0):
            raise ValueError(
                f"BACKTESTER_RISK_FREE_RATE={self.BACKTESTER_RISK_FREE_RATE} — "
                f"must be a decimal in [0.0, 1.0) (e.g. 0.05 for 5%)"
            )

        if not (1 <= self.BACKTESTER_TRADING_DAYS <= 365):
            raise ValueError(
                f"BACKTESTER_TRADING_DAYS={self.BACKTESTER_TRADING_DAYS} — "
                f"must be between 1 and 365"
            )

    @property
    def BACKTESTER_LOG_LEVEL_INT(self) -> int:
        """Return the logging level as an ``int`` usable by the stdlib."""
        return getattr(logging, self.BACKTESTER_LOG_LEVEL, logging.INFO)


settings = Settings()
