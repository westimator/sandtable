"""
src/sandtable/persistence/abstract_store.py

Abstract repository interface and RunSummary dataclass for persistence layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sandtable.config import BacktestConfig
    from sandtable.core.result import BacktestResult


@dataclass
class RunSummary:
    """
    Lightweight summary of a persisted backtest run.

    Returned by list_runs() for browsing without loading full results.
    """
    run_id: str
    strategy_name: str
    universe: list[str]
    start_date: str
    end_date: str
    sharpe_ratio: float
    cagr: float
    max_drawdown: float
    total_trades: int
    created_at: str
    tags: dict[str, str] = field(default_factory=dict)


class AbstractResultStore(ABC):
    """
    Abstract interface for persisting backtest results.

    Implementations can back onto SQLite, PostgreSQL, or any other store.
    """

    @abstractmethod
    def save_run(
        self,
        config: BacktestConfig,
        result: BacktestResult,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Persist a run. Returns run_id (UUID)."""

    @abstractmethod
    def load_run(self, run_id: str) -> tuple[BacktestConfig, BacktestResult]:
        """Load by ID. Raises KeyError if not found."""

    @abstractmethod
    def list_runs(
        self,
        strategy: str | None = None,
        after: str | None = None,
        before: str | None = None,
        min_sharpe: float | None = None,
        limit: int = 50,
    ) -> list[RunSummary]:
        """Query runs with optional filters."""

    @abstractmethod
    def delete_run(self, run_id: str) -> bool:
        """Delete by ID. Returns True if found and deleted."""
