from __future__ import annotations

import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

from causal_cell import CausalCell, DecisionStatus, SQLiteReplayStore
from tests.helpers import NOW, base_policy, base_proposal, rebound


def valid_claim() -> None:
    return None


class SQLiteReplayStoreTests(unittest.TestCase):
    def test_nonce_and_idempotency_claims_survive_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            database = Path(temp) / "state" / "causal-cell.sqlite3"
            first = SQLiteReplayStore(database)
            self.assertIsNone(
                first.consume(
                    "nonce-1",
                    "effect-1",
                    "sha256:first",
                    validate=valid_claim,
                )
            )

            reopened = SQLiteReplayStore(database)
            self.assertEqual(
                "INTENT_REPLAYED",
                reopened.consume(
                    "nonce-1",
                    "effect-2",
                    "sha256:second",
                    validate=valid_claim,
                ),
            )
            self.assertEqual(
                "IDEMPOTENCY_REPLAYED",
                reopened.consume(
                    "nonce-2",
                    "effect-1",
                    "sha256:second",
                    validate=valid_claim,
                ),
            )
            self.assertEqual("sha256:first", reopened.consumed_by("nonce-1"))
            self.assertEqual(
                "sha256:first",
                reopened.idempotency_consumed_by("effect-1"),
            )

    def test_nonce_claim_is_atomic_across_instances(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            database = Path(temp) / "state.sqlite3"
            stores = [SQLiteReplayStore(database) for _ in range(12)]
            with ThreadPoolExecutor(max_workers=12) as pool:
                outcomes = list(
                    pool.map(
                        lambda store: store.consume(
                            "shared-nonce",
                            "shared-effect",
                            "sha256:proposal",
                            validate=valid_claim,
                        ),
                        stores,
                    )
                )
            self.assertEqual(1, outcomes.count(None))
            self.assertEqual(11, outcomes.count("INTENT_REPLAYED"))

    def test_expiry_during_sqlite_lock_wait_does_not_poison_claim(self) -> None:
        class TracedSQLiteReplayStore(SQLiteReplayStore):
            def __init__(self, *args, **kwargs) -> None:
                self.begin_attempted = threading.Event()
                super().__init__(*args, **kwargs)

            def _connect(self):
                connection = super()._connect()
                connection.set_trace_callback(
                    lambda statement: (
                        self.begin_attempted.set()
                        if statement == "BEGIN IMMEDIATE"
                        else None
                    )
                )
                return connection

        class AdvancingClock:
            def __init__(self) -> None:
                self.first_read = threading.Event()
                self.calls = 0

            def __call__(self):
                self.calls += 1
                if self.calls == 1:
                    self.first_read.set()
                    return NOW
                return NOW.replace(hour=22, minute=1)

        with tempfile.TemporaryDirectory() as temp:
            database = Path(temp) / "state.sqlite3"
            store = TracedSQLiteReplayStore(database, busy_timeout_seconds=2)
            blocker = sqlite3.connect(database, isolation_level=None)
            blocker.execute("BEGIN IMMEDIATE")
            clock = AdvancingClock()
            calls = 0

            def executor(_proposal):
                nonlocal calls
                calls += 1
                return {"ok": True}

            try:
                with tempfile.TemporaryDirectory() as evidence:
                    cell = CausalCell(
                        base_policy(),
                        evidence,
                        nonce_store=store,
                        clock=clock,
                    )
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(cell.execute, base_proposal(), executor)
                        self.assertTrue(clock.first_read.wait(timeout=1))
                        self.assertTrue(store.begin_attempted.wait(timeout=1))
                        self.assertEqual(1, clock.calls)
                        blocker.commit()
                        run = future.result(timeout=2)
            finally:
                if blocker.in_transaction:
                    blocker.rollback()
                blocker.close()

            self.assertEqual(DecisionStatus.BLOCK, run.decision.status)
            self.assertIn("PROPOSAL_EXPIRED", run.decision.reasons)
            self.assertFalse(run.observation["executor_invoked"])
            self.assertEqual(0, calls)
            self.assertIsNone(store.consumed_by("nonce-001"))
            self.assertIsNone(store.idempotency_consumed_by("inspect-001"))

            retry = rebound(
                base_proposal(),
                attempt_id="attempt-after-expiry-race",
                expires_at="2026-08-27T23:00:00Z",
            )
            with tempfile.TemporaryDirectory() as retry_evidence:
                retry_run = CausalCell(
                    base_policy(),
                    retry_evidence,
                    nonce_store=store,
                    clock=clock,
                ).execute(retry, executor)
            self.assertEqual(DecisionStatus.ACCEPT, retry_run.decision.status)
            self.assertEqual(1, calls)

    def test_semantic_run_claim_is_durable_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            database = Path(temp) / "state.sqlite3"
            stores = [SQLiteReplayStore(database) for _ in range(12)]
            with ThreadPoolExecutor(max_workers=12) as pool:
                outcomes = list(
                    pool.map(
                        lambda store: store.consume_run("semantic-run"),
                        stores,
                    )
                )
            self.assertEqual(1, outcomes.count(True))
            self.assertEqual(11, outcomes.count(False))
            self.assertFalse(SQLiteReplayStore(database).consume_run("semantic-run"))

    def test_unknown_schema_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            database = Path(temp) / "state.sqlite3"
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    "CREATE TABLE causal_cell_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                connection.execute(
                    "INSERT INTO causal_cell_meta(key, value) VALUES (?, ?)",
                    ("schema_version", "999"),
                )
                connection.commit()
            with self.assertRaisesRegex(RuntimeError, "unsupported"):
                SQLiteReplayStore(database)


if __name__ == "__main__":
    unittest.main()
