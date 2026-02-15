"""
src/sandtable/utils/logger.py

Centralized logging configuration for the project.
"""

import logging

from sandtable.config import settings


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger by name."""
    logger = logging.getLogger(name)
    logger.setLevel(settings.BACKTESTER_LOG_LEVEL_INT)

    if not logger.handlers:
        handler = logging.StreamHandler()
        log_format = logging.Formatter(
            fmt=settings.BACKTESTER_LOG_FORMAT,
            datefmt=settings.BACKTESTER_LOG_DATE_FORMAT,
        )
        handler.setFormatter(fmt=log_format)
        logger.addHandler(handler)

    logger.propagate = False
    return logger
