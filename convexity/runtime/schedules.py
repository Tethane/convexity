"""Durable delivery of supplied exchange-session timers.

The journal is deliberately conservative: a crash during a callback leaves an
uncertain run requiring reconciliation. It never claims exactly-once external
side effects. Use the run key in durable order IDs before submitting orders.
"""

from collections.abc import Awaitable, Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sqlite3
from typing import Literal

from .types import Timer, aware
from .state import DeliveryResult


@dataclass(frozen=True)
class ScheduledRun:
    key: str
    timer: Timer
    delivered_at: datetime
    policy: Literal["skip", "catch_up"]

    def order_id(self, label: str) -> str:
        """Stable broker-safe intent ID; use a distinct label for each order."""
        if not label:
            raise ValueError("An order label is required.")
        return hashlib.sha256(json.dumps((self.key, label)).encode()).hexdigest()[:48]


class DurableSchedule:
    """Persist timer delivery; skip missed entries and catch up overdue exits.

    Construct once per process with its startup time. Timers before startup are
    skipped unless explicitly configured catch_up. Polls during the process
    deliver every due timer, irrespective of polling latency. Supply a distinct
    namespace for labeled one-shot runs. A namespace's schedule is immutable.
    """

    def __init__(
        self,
        path: str | Path,
        namespace: str,
        timers: Iterable[Timer],
        *,
        started_at: datetime,
        policy: Literal["skip", "catch_up"] = "skip",
    ):
        aware(started_at)
        if not namespace or policy not in ("skip", "catch_up"):
            raise ValueError("Supply a namespace and skip/catch_up policy.")
        self.started_at = started_at.astimezone(timezone.utc)
        self.namespace = namespace
        self.policy: Literal["skip", "catch_up"] = policy
        self.timers = tuple(sorted(set(timers), key=lambda t: (t.time, t.name)))
        self._last_poll = self.started_at
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("""CREATE TABLE IF NOT EXISTS schedule_runs (
            namespace TEXT NOT NULL, name TEXT NOT NULL, due TEXT NOT NULL,
            policy TEXT NOT NULL, status TEXT NOT NULL, delivered TEXT,
            detail TEXT NOT NULL DEFAULT '', PRIMARY KEY(namespace,name,due))""")
        with self._db:
            existing = self._db.execute(
                "SELECT name,due,policy FROM schedule_runs WHERE namespace=?",
                (namespace,),
            ).fetchall()
            spec = {(t.name, t.time.isoformat(), policy) for t in self.timers}
            if (
                existing
                and {(r["name"], r["due"], r["policy"]) for r in existing} != spec
            ):
                raise ValueError(
                    "Schedule namespace already has a different specification."
                )
            for timer in self.timers:
                self._db.execute(
                    "INSERT OR IGNORE INTO schedule_runs(namespace,name,due,policy,status) VALUES(?,?,?,?,?)",
                    (namespace, timer.name, timer.time.isoformat(), policy, "pending"),
                )

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self) -> None:
        self._db.close()

    def status(self) -> tuple[dict, ...]:
        return tuple(
            dict(r)
            for r in self._db.execute(
                "SELECT * FROM schedule_runs WHERE namespace=? ORDER BY due,name",
                (self.namespace,),
            )
        )

    def _due(self, now: datetime, limit: int | None = None) -> Iterator[ScheduledRun]:
        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError("limit must be a positive integer or None.")
        processed = 0
        aware(now)
        now = now.astimezone(timezone.utc)
        if now < self._last_poll:
            raise ValueError("Schedule clock cannot move backward.")
        self._last_poll = now
        if any(r["status"] == "uncertain" for r in self.status()):
            raise RuntimeError(
                "An interrupted schedule needs reconciliation before delivery resumes."
            )
        for timer in self.timers:
            if timer.time > now:
                break
            identity = (self.namespace, timer.name, timer.time.isoformat())
            with self._db:
                self._db.execute("BEGIN IMMEDIATE")
                uncertain = self._db.execute(
                    "SELECT 1 FROM schedule_runs WHERE namespace=? AND status='uncertain'",
                    (self.namespace,),
                ).fetchone()
                if uncertain:
                    raise RuntimeError(
                        "An interrupted schedule needs reconciliation before delivery resumes."
                    )
                row = self._db.execute(
                    "SELECT status FROM schedule_runs WHERE namespace=? AND name=? AND due=?",
                    identity,
                ).fetchone()
                if row["status"] != "pending":
                    continue
                missed = timer.time < self.started_at and self.policy == "skip"
                self._db.execute(
                    "UPDATE schedule_runs SET status=?,delivered=? WHERE namespace=? AND name=? AND due=?",
                    ("skipped" if missed else "uncertain", now.isoformat(), *identity),
                )
            processed += 1
            if missed:
                if limit is not None and processed >= limit:
                    return
                continue
            # Persist claim before invoking. Exceptions/crashes retain uncertainty.
            key = hashlib.sha256(
                json.dumps(identity, separators=(",", ":")).encode()
            ).hexdigest()
            yield ScheduledRun(key, timer, now, self.policy)
            if limit is not None and processed >= limit:
                return

    def _complete(self, run: ScheduledRun) -> None:
        with self._db:
            self._db.execute(
                "UPDATE schedule_runs SET status='completed' WHERE namespace=? AND name=? AND due=?",
                (self.namespace, run.timer.name, run.timer.time.isoformat()),
            )

    def _failed(self, run: ScheduledRun, error: BaseException) -> None:
        with self._db:
            self._db.execute(
                "UPDATE schedule_runs SET detail=? WHERE namespace=? AND name=? AND due=?",
                (
                    f"{type(error).__name__}: {error}",
                    self.namespace,
                    run.timer.name,
                    run.timer.time.isoformat(),
                ),
            )

    def poll(
        self, now: datetime, callback: Callable[[ScheduledRun], None | DeliveryResult]
    ) -> tuple[ScheduledRun, ...]:
        delivered = []
        for run in self._due(now):
            try:
                result = callback(run)
                if isinstance(result, DeliveryResult):
                    result.raise_for_failure()
            except BaseException as error:
                self._failed(run, error)
                raise
            self._complete(run)
            delivered.append(run)
        return tuple(delivered)

    async def poll_async(
        self,
        now: datetime,
        callback: Callable[[ScheduledRun], Awaitable[None | DeliveryResult]],
        *,
        limit: int | None = None,
    ) -> tuple[ScheduledRun, ...]:
        """Await delivery; limit counts pending rows processed, including missed skips."""
        delivered = []
        for run in self._due(now, limit):
            try:
                result = await callback(run)
                if isinstance(result, DeliveryResult):
                    result.raise_for_failure()
            except BaseException as error:
                self._failed(run, error)
                raise
            self._complete(run)
            delivered.append(run)
        return tuple(delivered)

    def resolve(self, timer: Timer, *, completed: bool, detail: str) -> None:
        """Record an operator's reconciled outcome; never automatically retry."""
        if not detail.strip():
            raise ValueError("Reconciliation evidence is required.")
        with self._db:
            cursor = self._db.execute(
                """UPDATE schedule_runs SET status=?,detail=?
                WHERE namespace=? AND name=? AND due=? AND status='uncertain'""",
                (
                    "completed" if completed else "skipped",
                    detail,
                    self.namespace,
                    timer.name,
                    timer.time.isoformat(),
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("Only an uncertain scheduled run can be reconciled.")
