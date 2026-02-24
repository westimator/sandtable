"""
Bidirectional synchronization between two SQL-backed result stores.

Provides :func:`sync_stores`, which compares two
:class:`~sandtable.persistence.abstract_sql_result_store.AbstractSQLResultStore`
instances by run ID and copies missing runs in both directions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sandtable.persistence.abstract_sql_result_store import AbstractSQLResultStore
from sandtable.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SyncResult:
    """
    Summary of a bidirectional sync operation.

    Attributes:
        copied_a_to_b: number of runs copied from store A to store B.
        copied_b_to_a: number of runs copied from store B to store A.
        failed_a_to_b: run IDs that failed to copy from A to B.
        failed_b_to_a: run IDs that failed to copy from B to A.
    """
    copied_a_to_b: int = 0
    copied_b_to_a: int = 0
    failed_a_to_b: list[str] = field(default_factory=list)
    failed_b_to_a: list[str] = field(default_factory=list)

    @property
    def total_copied(self) -> int:
        return self.copied_a_to_b + self.copied_b_to_a

    @property
    def total_failed(self) -> int:
        return len(self.failed_a_to_b) + len(self.failed_b_to_a)


def _load_run_metadata(store: AbstractSQLResultStore, run_id: str) -> dict[str, Any]:
    """
    Load tags and created_at for a single run.

    Parameters:
        store: the result store to read from.
        run_id: UUID of the run.

    Returns:
        dict with keys ``'tags'`` (dict[str, str]) and ``'created_at'``
        (str or None).
    """
    p = store._placeholder
    conn = store._connect()
    try:
        cur = conn.cursor()

        # get created_at
        cur.execute(f"SELECT created_at FROM runs WHERE run_id = {p}", (run_id,))
        row = cur.fetchone()
        created_at = row[0] if row else None

        # get tags
        cur.execute(
            f"SELECT {store._key_col}, value FROM tags WHERE run_id = {p}",
            (run_id,),
        )
        tags = {r[0]: r[1] for r in cur.fetchall()}

        return {"created_at": created_at, "tags": tags}
    finally:
        cur.close()
        conn.close()


def _copy_runs(
    source: AbstractSQLResultStore,
    dest: AbstractSQLResultStore,
    run_ids: set[str],
) -> tuple[int, list[str]]:
    """
    Copy runs from *source* to *dest*, preserving run_id and created_at.

    Each run is copied independently so that a single failure does not
    abort the remaining transfers.

    Parameters:
        source: store to read runs from.
        dest: store to write runs into.
        run_ids: set of run UUIDs to transfer.

    Returns:
        Tuple of (copied_count, failed_run_ids).
    """
    copied = 0
    failed: list[str] = []

    for run_id in sorted(run_ids):
        try:
            config, result = source.load_run(run_id)
            meta = _load_run_metadata(source, run_id)
            dest._insert_run(
                run_id=run_id,
                config=config,
                result=result,
                tags=meta["tags"] or None,
                created_at=meta["created_at"],
            )
            copied += 1
            logger.debug("Copied run %s", run_id)
        except Exception as exc:
            # duplicate key means the run already exists in dest - skip
            exc_msg = str(exc).lower()
            if "duplicate" in exc_msg or "unique constraint" in exc_msg:
                logger.info("Run %s already exists in destination, skipping", run_id)
                continue
            logger.exception("Failed to copy run %s", run_id)
            failed.append(run_id)

    return copied, failed


def sync_stores(
    store_a: AbstractSQLResultStore,
    store_b: AbstractSQLResultStore,
) -> SyncResult:
    """
    Bidirectional sync between two SQL result stores.

    Compares run IDs in each store and copies any runs that are present
    in one but missing from the other. Original run_ids and created_at
    timestamps are preserved so that synced runs are identical on both
    sides.

    Typical use case is keeping a local SQLite store in sync with a
    remote MySQL store::

        sync_stores(sqlite_store, mysql_store)

    Parameters:
        store_a: first result store.
        store_b: second result store.

    Returns:
        A :class:`SyncResult` summarizing how many runs were copied in
        each direction and which (if any) failed.

    Raises:
        No exceptions are raised for individual run failures - they are
        captured in :attr:`SyncResult.failed_a_to_b` and
        :attr:`SyncResult.failed_b_to_a`.
    """
    ids_a = store_a.list_run_ids()
    ids_b = store_b.list_run_ids()

    missing_in_b = ids_a - ids_b
    missing_in_a = ids_b - ids_a

    name_a = type(store_a).__name__
    name_b = type(store_b).__name__
    logger.info(
        "Sync: %d runs in %s, %d in %s, %d missing in %s, %d missing in %s",
        len(ids_a), name_a, len(ids_b), name_b, len(missing_in_b), name_b, len(missing_in_a), name_a,
    )

    result = SyncResult()

    if missing_in_b:
        result.copied_a_to_b, result.failed_a_to_b = _copy_runs(store_a, store_b, missing_in_b)

    if missing_in_a:
        result.copied_b_to_a, result.failed_b_to_a = _copy_runs(store_b, store_a, missing_in_a)

    logger.info(
        "Sync complete: %d copied (%d %s->%s, %d %s->%s), %d failed",
        result.total_copied, result.copied_a_to_b, name_a, name_b,
        result.copied_b_to_a, name_b, name_a, result.total_failed,
    )

    return result
