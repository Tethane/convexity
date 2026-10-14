"""Durable strategy checkpoints and conservative callback recovery.

A callback is claimed before user code and completed only after commands finish.
A crash in between requires reconciliation, even when a checkpoint was saved.
No claim of exactly-once arbitrary Python or external effects is made.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3


class DeliveryError(RuntimeError):
    """A strategy delivery failed or has an unresolved outcome."""


@dataclass(frozen=True)
class DeliveryResult:
    paused: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def successful(self) -> bool:
        return not self.paused

    def raise_for_failure(self) -> None:
        if not self.successful:
            raise DeliveryError(
                "Strategy delivery failed or remains unresolved: "
                + ", ".join(self.paused)
                + ("; " + "; ".join(self.errors) if self.errors else "")
            )


class StrategyState:
    """Account-scoped JSON checkpoints in SQLite, safe to share with OrderJournal.

    Strategies explicitly implement dump_state/load_state; the default is None.
    Resolve an uncertain callback only after reconciling broker/journal effects.
    Resolution never retries a callback; a restarted strategy resumes its saved
    state on future events. A changed strategy class needs a new owner identity.
    """

    def __init__(self, path: str | Path, account: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.account = account
        self._db = sqlite3.connect(path)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS strategy_checkpoints (
                account TEXT NOT NULL, owner TEXT NOT NULL, kind TEXT NOT NULL,
                state TEXT NOT NULL, PRIMARY KEY(account,owner));
            CREATE TABLE IF NOT EXISTS strategy_subscriptions (
                account TEXT NOT NULL, owner TEXT NOT NULL, state TEXT NOT NULL,
                PRIMARY KEY(account,owner));
            CREATE TABLE IF NOT EXISTS strategy_callbacks (
                account TEXT NOT NULL, owner TEXT NOT NULL, identity TEXT NOT NULL,
                status TEXT NOT NULL, detail TEXT NOT NULL DEFAULT '',
                PRIMARY KEY(account,owner,identity));
        """)

    def close(self) -> None:
        self._db.close()

    def load(self, owner: str, kind: str):
        row = self._db.execute(
            "SELECT kind,state FROM strategy_checkpoints WHERE account=? AND owner=?",
            (self.account, owner),
        ).fetchone()
        if row is None:
            return None
        if row["kind"] != kind:
            raise DeliveryError(
                f"Strategy {owner} checkpoint belongs to a different class; use a new owner or migrate it."
            )
        return json.loads(row["state"])

    def has_checkpoint(self, owner: str) -> bool:
        return (
            self._db.execute(
                "SELECT 1 FROM strategy_checkpoints WHERE account=? AND owner=?",
                (self.account, owner),
            ).fetchone()
            is not None
        )

    def consumed(self, owner: str, identity: str) -> bool:
        """Whether a callback was completed/baselined or explicitly reconciled."""
        row = self._db.execute(
            """SELECT status FROM strategy_callbacks
            WHERE account=? AND owner=? AND identity=?""",
            (self.account, owner, identity),
        ).fetchone()
        return row is not None and row["status"] in ("completed", "resolved")

    def baseline(self, owner: str, identities) -> None:
        """Adopt pre-deployment broker history without replaying historical callbacks."""
        with self._db:
            self._db.executemany(
                """INSERT OR IGNORE INTO strategy_callbacks VALUES(?,?,?,'completed',
                'Historical baseline at initial strategy deployment')""",
                ((self.account, owner, identity) for identity in identities),
            )

    def save(self, owner: str, kind: str, state) -> None:
        encoded = json.dumps(
            state, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
        with self._db:
            self._db.execute(
                """INSERT INTO strategy_checkpoints VALUES(?,?,?,?)
                ON CONFLICT(account,owner) DO UPDATE SET kind=excluded.kind,state=excluded.state""",
                (self.account, owner, kind, encoded),
            )

    def load_subscriptions(self, owner: str) -> dict | None:
        row = self._db.execute(
            "SELECT state FROM strategy_subscriptions WHERE account=? AND owner=?",
            (self.account, owner),
        ).fetchone()
        return json.loads(row["state"]) if row is not None else None

    def save_subscriptions(self, owner: str, state: dict) -> None:
        encoded = json.dumps(
            state, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
        with self._db:
            self._db.execute(
                """INSERT INTO strategy_subscriptions VALUES(?,?,?)
                ON CONFLICT(account,owner) DO UPDATE SET state=excluded.state""",
                (self.account, owner, encoded),
            )

    def unresolved(self, owner: str) -> tuple[dict, ...]:
        return tuple(
            dict(row)
            for row in self._db.execute(
                """SELECT identity,status,detail FROM strategy_callbacks
            WHERE account=? AND owner=? AND status='uncertain'""",
                (self.account, owner),
            )
        )

    def claim(self, owner: str, identity: str) -> bool:
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            if self.unresolved(owner):
                raise DeliveryError(
                    f"Strategy {owner} has an interrupted callback requiring reconciliation."
                )
            cursor = self._db.execute(
                "INSERT OR IGNORE INTO strategy_callbacks VALUES(?,?,?,'uncertain','')",
                (self.account, owner, identity),
            )
            return cursor.rowcount == 1

    def complete(self, owner: str, identity: str) -> None:
        with self._db:
            self._db.execute(
                """UPDATE strategy_callbacks SET status='completed'
                WHERE account=? AND owner=? AND identity=?""",
                (self.account, owner, identity),
            )
            kind = identity.split(":", 1)[0]
            if kind in ("StockQuotes", "AccountUpdate", "ConnectionHealth"):
                # Observation callbacks are replaceable current snapshots, not
                # trading-event replay identities. Keep a bounded recent audit;
                # uncertain/resolved outcomes and all other kinds are durable.
                self._db.execute(
                    """DELETE FROM strategy_callbacks WHERE rowid IN (
                    SELECT rowid FROM strategy_callbacks WHERE account=? AND owner=?
                    AND status='completed' AND identity>=? AND identity<?
                    ORDER BY rowid DESC LIMIT -1 OFFSET 100)""",
                    (self.account, owner, kind + ":", kind + ";"),
                )

    def fail(self, owner: str, identity: str, detail: str) -> None:
        with self._db:
            self._db.execute(
                """UPDATE strategy_callbacks SET detail=?
                WHERE account=? AND owner=? AND identity=? AND status='uncertain'""",
                (detail, self.account, owner, identity),
            )

    def resolve(self, owner: str, identity: str, *, detail: str) -> None:
        """Record reconciled effects; keep the checkpoint and never replay the callback."""
        if not detail.strip():
            raise ValueError("Reconciliation evidence is required.")
        with self._db:
            cursor = self._db.execute(
                """UPDATE strategy_callbacks SET status='resolved',detail=?
                WHERE account=? AND owner=? AND identity=? AND status='uncertain'""",
                (detail, self.account, owner, identity),
            )
            if cursor.rowcount != 1:
                raise ValueError("Only an uncertain callback can be reconciled.")
