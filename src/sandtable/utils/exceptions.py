"""
src/sandtable/utils/exceptions.py

Shared exception classes used across the sandtable package.
"""


class DataFetchError(Exception):
    """Raised when a data provider fails to fetch data."""


class ConfigurationError(ValueError):
    """Raised when a configuration value is invalid or missing."""


class InsufficientDataError(ValueError):
    """Raised when there is not enough data for the requested operation."""


class RunNotFoundError(KeyError):
    """Raised when a persisted run cannot be found by ID."""


class StrategyValidationError(ValueError):
    """Raised when strategy parameters fail validation."""
