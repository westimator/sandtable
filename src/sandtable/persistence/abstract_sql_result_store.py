"""
src/sandtable/persistence/abstract_sql_result_store.py

Shared implementation for SQL-backed result stores.

Subclasses provide connection and dialect details; this class handles
serialization, deserialization, and query logic common to all SQL backends.
"""

from __future__ import annotations

import json
import os
import time
from abc import abstractmethod
from datetime import datetime, timezone
from typing import Any

from sandtable.config import BacktestConfig
from sandtable.core.events import Direction, FillEvent, RiskBreachEvent
from sandtable.core.result import BacktestResult
from sandtable.metrics.performance import calculate_metrics
from sandtable.persistence.abstract_store import AbstractResultStore, RunSummary
from sandtable.portfolio.portfolio import EquityPoint
from sandtable.utils.exceptions import RunNotFoundError
from sandtable.utils.logger import get_logger

logger = get_logger(__name__)


class AbstractSQLResultStore(AbstractResultStore):
    """
    Base class for SQL-backed result stores (SQLite, MySQL, etc.).

    Implements save_run, load_run, list_runs, and delete_run using standard
    SQL. Subclasses supply connection handling and dialect-specific details
    via the abstract properties and methods below.
    """

    ## dialect hooks (subclasses must implement)

    @property
    @abstractmethod
    def _placeholder(self) -> str:
        """
        Parameter placeholder character.

        '?' for SQLite, '%s' for MySQL.
        """

    @property
    @abstractmethod
    def _key_col(self) -> str:
        """
        Quoted column name for the tags.key column.

        'key' for SQLite, '`key`' for MySQL (reserved word).
        """

    @abstractmethod
    def _connect(self) -> Any:
        """
        Open a new database connection.
        """

    @abstractmethod
    def _dict_cursor(self, conn: Any) -> Any:
        """
        Return a cursor whose rows behave like dicts (access by column name).
        """

    @abstractmethod
    def _ensure_schema(self) -> None:
        """
        Create tables and indexes if they don't exist. Idempotent.
        """

    ## helpers

    @staticmethod
    def _parse_json(value: str | Any) -> Any:
        """
        Parse a JSON column value.

        MySQL JSON columns may return a dict directly; SQLite always returns str.
        """
        return json.loads(value) if isinstance(value, str) else value

    ## shared CRUD

    def save_run(
        self,
        config: BacktestConfig,
        result: BacktestResult,
        tags: dict[str, str] | None = None,
    ) -> str:
        """
        Persist a backtest run.

        Returns the generated run_id.
        """
        run_id = f"{int(time.time() * 1_000)}_{os.urandom(4).hex()}"
        now = datetime.now(tz=timezone.utc).isoformat()
        self._insert_run(run_id, config, result, tags, now)
        return run_id

    def _insert_run(
        self,
        run_id: str,
        config: BacktestConfig,
        result: BacktestResult,
        tags: dict[str, str] | None,
        created_at: str,
    ) -> None:
        """
        Low-level insert of a run with all child rows.

        Used by save_run() and by sync logic that must preserve original
        run_id and created_at values.
        """
        logger.debug("Inserting run %s", run_id)

        p = self._placeholder
        m = result.metrics
        start_str = result.start_date.isoformat() if result.start_date else None
        end_str = result.end_date.isoformat() if result.end_date else None

        conn = self._connect()
        try:
            cur = conn.cursor()

            # insert run
            cur.execute(
                f"""INSERT INTO runs (
                    run_id, strategy_name, config_json, universe,
                    start_date, end_date, initial_capital,
                    sharpe_ratio, sortino_ratio, cagr, max_drawdown,
                    total_return, total_trades, win_rate, profit_factor,
                    created_at
                ) VALUES ({', '.join([p] * 16)})""",
                (
                    run_id,
                    config.strategy_cls.__name__,
                    json.dumps(config.to_dict()),
                    json.dumps(config.universe),
                    start_str,
                    end_str,
                    config.initial_capital,
                    m.sharpe_ratio,
                    m.sortino_ratio,
                    m.cagr,
                    m.max_drawdown,
                    m.total_return,
                    m.num_trades,
                    m.win_rate,
                    m.profit_factor,
                    created_at,
                ),
            )

            # insert equity curve (deduplicate: keep last entry per timestamp)
            if result.equity_curve:
                deduped: dict[str, EquityPoint] = {}
                for ep in result.equity_curve:
                    deduped[ep.timestamp.isoformat()] = ep
                cur.executemany(
                    f"""INSERT INTO equity_curves
                       (run_id, timestamp, equity, cash, positions_value)
                       VALUES ({', '.join([p] * 5)})""",
                    [
                        (run_id, ts, ep.equity, ep.cash, ep.positions_value)
                        for ts, ep in deduped.items()
                    ],
                )

            # insert fills
            if result.trades:
                cur.executemany(
                    f"""INSERT INTO fills
                       (run_id, fill_index, symbol, timestamp, direction,
                        quantity, fill_price, commission, slippage, market_impact)
                       VALUES ({', '.join([p] * 10)})""",
                    [
                        (
                            run_id, i, fill.symbol,
                            fill.timestamp.isoformat(), fill.direction.name,
                            fill.quantity, fill.fill_price, fill.commission,
                            fill.slippage, fill.market_impact,
                        )
                        for i, fill in enumerate(result.trades)
                    ],
                )

            # insert risk breaches
            risk_breaches = result.parameters.get("risk_breaches", [])
            if risk_breaches:
                cur.executemany(
                    f"""INSERT INTO risk_breaches
                       (run_id, breach_index, timestamp, rule_name, symbol,
                        proposed_qty, action, breach_value, threshold, final_qty)
                       VALUES ({', '.join([p] * 10)})""",
                    [
                        (
                            run_id, i,
                            rb.timestamp.isoformat() if isinstance(rb, RiskBreachEvent) else rb["timestamp"],
                            rb.rule_name if isinstance(rb, RiskBreachEvent) else rb["rule_name"],
                            rb.symbol if isinstance(rb, RiskBreachEvent) else rb["symbol"],
                            rb.proposed_qty if isinstance(rb, RiskBreachEvent) else rb["proposed_qty"],
                            rb.action if isinstance(rb, RiskBreachEvent) else rb["action"],
                            rb.breach_value if isinstance(rb, RiskBreachEvent) else rb["breach_value"],
                            rb.threshold if isinstance(rb, RiskBreachEvent) else rb["threshold"],
                            rb.final_qty if isinstance(rb, RiskBreachEvent) else rb.get("final_qty"),
                        )
                        for i, rb in enumerate(risk_breaches)
                    ],
                )

            # insert tags
            if tags:
                cur.executemany(
                    f"INSERT INTO tags (run_id, {self._key_col}, value) VALUES ({', '.join([p] * 3)})",
                    [(run_id, k, v) for k, v in tags.items()],
                )

            conn.commit()
        finally:
            cur.close()
            conn.close()

    def list_run_ids(self) -> set[str]:
        """
        Return the set of all run_ids in the store.

        Lightweight query used for sync diffing.
        """
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute("SELECT run_id FROM runs")
            return {row[0] for row in cur.fetchall()}
        finally:
            cur.close()
            conn.close()

    def load_run(self, run_id: str) -> tuple[BacktestConfig, BacktestResult]:
        """
        Load a run by ID.

        Raises RunNotFoundError if not found.
        """
        logger.debug("Loading run %s", run_id)
        p = self._placeholder
        conn = self._connect()
        try:
            cur = self._dict_cursor(conn)

            # load run metadata
            cur.execute(f"SELECT * FROM runs WHERE run_id = {p}", (run_id,))
            row = cur.fetchone()
            if row is None:
                raise RunNotFoundError(run_id)

            config = BacktestConfig.from_dict(self._parse_json(row["config_json"]))
            universe_list = self._parse_json(row["universe"])

            # load equity curve
            cur.execute(
                f"SELECT timestamp, equity, cash, positions_value "
                f"FROM equity_curves WHERE run_id = {p} ORDER BY timestamp",
                (run_id,),
            )
            equity_curve = [
                EquityPoint(
                    timestamp=datetime.fromisoformat(r["timestamp"]),
                    equity=r["equity"],
                    cash=r["cash"],
                    positions_value=r["positions_value"],
                )
                for r in cur.fetchall()
            ]

            # load fills
            cur.execute(
                f"SELECT symbol, timestamp, direction, quantity, fill_price, "
                f"commission, slippage, market_impact "
                f"FROM fills WHERE run_id = {p} ORDER BY fill_index",
                (run_id,),
            )
            trades = [
                FillEvent(
                    timestamp=datetime.fromisoformat(r["timestamp"]),
                    symbol=r["symbol"],
                    direction=Direction[r["direction"]],
                    quantity=r["quantity"],
                    fill_price=r["fill_price"],
                    commission=r["commission"],
                    slippage=r["slippage"],
                    market_impact=r["market_impact"],
                )
                for r in cur.fetchall()
            ]

            # load risk breaches
            cur.execute(
                f"SELECT timestamp, rule_name, symbol, proposed_qty, action, "
                f"breach_value, threshold, final_qty "
                f"FROM risk_breaches WHERE run_id = {p} ORDER BY breach_index",
                (run_id,),
            )
            risk_breaches = [
                RiskBreachEvent(
                    timestamp=datetime.fromisoformat(r["timestamp"]),
                    rule_name=r["rule_name"],
                    symbol=r["symbol"],
                    proposed_qty=r["proposed_qty"],
                    action=r["action"],
                    breach_value=r["breach_value"],
                    threshold=r["threshold"],
                    final_qty=r["final_qty"],
                )
                for r in cur.fetchall()
            ]

            # reconstruct metrics from equity curve and trades
            metrics = calculate_metrics(equity_curve, trades)

            # reconstruct parameters
            parameters: dict[str, Any] = {
                "strategy": row["strategy_name"],
                "symbols": universe_list,
                "initial_capital": row["initial_capital"],
            }
            if risk_breaches:
                parameters["risk_breaches"] = risk_breaches

            start_date = datetime.fromisoformat(row["start_date"]) if row["start_date"] else None
            end_date = datetime.fromisoformat(row["end_date"]) if row["end_date"] else None

            result = BacktestResult(
                metrics=metrics,
                equity_curve=equity_curve,
                trades=trades,
                parameters=parameters,
                price_data={},
                symbols=universe_list,
                initial_capital=row["initial_capital"],
                start_date=start_date,
                end_date=end_date,
            )

            logger.debug(
                "Loaded run %s: %d equity points, %d fills, %d breaches",
                run_id, len(equity_curve), len(trades), len(risk_breaches),
            )
            return config, result
        finally:
            cur.close()
            conn.close()

    def list_runs(
        self,
        strategy: str | None = None,
        after: str | None = None,
        before: str | None = None,
        min_sharpe: float | None = None,
        limit: int = 50,
    ) -> list[RunSummary]:
        """
        Query runs with optional filters.
        """
        logger.debug("Listing runs (strategy=%s, limit=%d)", strategy, limit)
        p = self._placeholder
        clauses: list[str] = []
        params: list[Any] = []

        if strategy is not None:
            clauses.append(f"strategy_name = {p}")
            params.append(strategy)
        if after is not None:
            clauses.append(f"created_at >= {p}")
            params.append(after)
        if before is not None:
            clauses.append(f"created_at <= {p}")
            params.append(before)
        if min_sharpe is not None:
            clauses.append(f"sharpe_ratio >= {p}")
            params.append(min_sharpe)

        where = ""
        if clauses:
            where = "WHERE " + " AND ".join(clauses)

        sql = f"SELECT * FROM runs {where} ORDER BY created_at DESC LIMIT {p}"
        params.append(limit)

        conn = self._connect()
        try:
            cur = self._dict_cursor(conn)
            cur.execute(sql, params)
            rows = cur.fetchall()

            summaries: list[RunSummary] = []
            for row in rows:
                cur.execute(
                    f"SELECT {self._key_col}, value FROM tags WHERE run_id = {p}",
                    (row["run_id"],),
                )
                tag_rows = cur.fetchall()
                run_tags = {r["key"]: r["value"] for r in tag_rows}

                summaries.append(
                    RunSummary(
                        run_id=row["run_id"],
                        strategy_name=row["strategy_name"],
                        universe=self._parse_json(row["universe"]),
                        start_date=row["start_date"] or "",
                        end_date=row["end_date"] or "",
                        sharpe_ratio=row["sharpe_ratio"] or 0.0,
                        cagr=row["cagr"] or 0.0,
                        max_drawdown=row["max_drawdown"] or 0.0,
                        total_trades=row["total_trades"] or 0,
                        created_at=row["created_at"],
                        tags=run_tags,
                    )
                )

            return summaries
        finally:
            cur.close()
            conn.close()

    def delete_run(self, run_id: str) -> bool:
        """
        Delete a run by ID.

        Returns True if the run was found and deleted.
        """
        logger.debug("Deleting run %s", run_id)
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                f"DELETE FROM runs WHERE run_id = {self._placeholder}",
                (run_id,),
            )
            conn.commit()
            deleted = cur.rowcount > 0
            logger.debug(
                "Delete run %s: %s",
                run_id, "found" if deleted else "not found",
            )
            return deleted
        finally:
            cur.close()
            conn.close()
