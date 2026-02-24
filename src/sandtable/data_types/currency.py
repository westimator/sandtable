"""
src/sandtable/data_types/currency.py

Settlement currencies for tradeable instruments.
"""

from __future__ import annotations

from enum import StrEnum


class Currency(StrEnum):
    """
    Settlement currency for an instrument.
    """
    USD = "USD"
