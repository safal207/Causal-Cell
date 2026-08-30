"""Durable, process-safe replay state backed by the Python SQLite runtime."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = "1"


class SQLiteReplayStore:
    """One durable store for action nonces and Organism semantic-run keys.

    Every claim is made under ``BEGIN IMMEDIATE``.  Separate processes therefore
    serialize the check-and-insert boundary instead of merely sharing a file.
    """

    def __init__(
        self,
        database: str | Path,
        *,
        busy_timeout_seconds: float = 5.0,
    ) -> None:
        if (
            isinstance(busy_timeout_seconds, bool)
            or not isinstance(busy_timeout_seconds, (int, float))
            or busy_timeout_seconds <= 0
        ):
            raise ValueError("invalid SQLite busy timeout")
        self._path = Path(database)
        if self._path.exists() and not self._path.is_file():
            raise ValueError("SQLite database path must be a file")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._busy_timeout_ms = int(float(busy_timeout_seconds) * 1000)
        self._initialize()

    @property
    def database_path(self) -> Path:
        return self._path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._path,
            timeout=self._busy_timeout_ms / 1000,
            isolation_level=None,
        )
        connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS causal_cell_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO causal_cell_meta(key, value) VALUES (?, ?)",
                ("schema_version", SCHEMA_VERSION),
            )
            row = connection.execute(
                "SELECT value FROM causal_cell_meta WHERE key = ?",
                ("schema_version",),
            ).fetchone()
            if row != (SCHEMA_VERSION,):
                raise RuntimeError("unsupported SQLite replay-store schema")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS nonce_consumptions (
                    nonce TEXT PRIMARY KEY,
                    bound_proposal_digest TEXT NOT NULL,
                    consumed_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS idempotency_consumptions (
                    idempotency_key TEXT PRIMARY KEY,
                    bound_proposal_digest TEXT NOT NULL,
                    consumed_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS organism_runs (
                    semantic_run_key TEXT PRIMARY KEY,
                    consumed_at TEXT NOT NULL
                )
                """
            )

    def consume(
        self,
        nonce: str,
        idempotency_key: str,
        bound_proposal_digest: str,
        *,
        validate: Callable[[], str | None],
    ) -> str | None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM nonce_consumptions WHERE nonce = ?",
                (nonce,),
            ).fetchone():
                connection.rollback()
                return "INTENT_REPLAYED"
            if connection.execute(
                "SELECT 1 FROM idempotency_consumptions WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone():
                connection.rollback()
                return "IDEMPOTENCY_REPLAYED"
            validation_reason = validate()
            if validation_reason is not None:
                connection.rollback()
                return validation_reason
            consumed_at = datetime.now(UTC).isoformat()
            connection.execute(
                """
                INSERT INTO nonce_consumptions(nonce, bound_proposal_digest, consumed_at)
                VALUES (?, ?, ?)
                """,
                (nonce, bound_proposal_digest, consumed_at),
            )
            connection.execute(
                """
                INSERT INTO idempotency_consumptions(
                    idempotency_key, bound_proposal_digest, consumed_at
                ) VALUES (?, ?, ?)
                """,
                (idempotency_key, bound_proposal_digest, consumed_at),
            )
            connection.commit()
            return None
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def consumed_by(self, nonce: str) -> str | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT bound_proposal_digest FROM nonce_consumptions WHERE nonce = ?",
                (nonce,),
            ).fetchone()
        return None if row is None else row[0]

    def idempotency_consumed_by(self, idempotency_key: str) -> str | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT bound_proposal_digest
                FROM idempotency_consumptions
                WHERE idempotency_key = ?
                """,
                (idempotency_key,),
            ).fetchone()
        return None if row is None else row[0]

    def consume_run(self, semantic_run_key: str) -> bool:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO organism_runs(semantic_run_key, consumed_at)
                VALUES (?, ?)
                """,
                (semantic_run_key, datetime.now(UTC).isoformat()),
            )
            consumed = cursor.rowcount == 1
            connection.commit()
            return consumed
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
