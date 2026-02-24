"""
src/sandtable/data_types/instrument_type.py

Asset class classification for tradeable instruments.
"""

from __future__ import annotations

from enum import StrEnum


class InstrumentType(StrEnum):
    """
    Asset class of a tradeable instrument.

    Determines default tick size, margin, and multiplier behaviour.
    EQUITY is fully implemented; FUTURE and FX are stubbed for extension.

    Values:
        EQUITY: Shares of stock (tick=0.01, multiplier=1, margin=1.0)
        FUTURE: Exchange-traded futures contract (variable tick/multiplier, fractional margin)
        FX: Foreign exchange spot or forward (not yet implemented)
    """
    EQUITY = "EQUITY"
    FUTURE = "FUTURE"
    FX = "FX"
