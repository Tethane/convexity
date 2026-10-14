"""Persistent security identity and bitemporal symbol mappings.

Identifiers describe securities, not issuers or ticker spellings. Callers must
supply verified identifiers; there is deliberately no fuzzy identity matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from .core import DateLike, timestamp


class UniverseError(ValueError):
    """Missing, ambiguous, or historically unavailable reference data."""


class KnowledgeUnavailableError(UniverseError):
    """Reference data was not established by the decision timestamp."""


@dataclass(frozen=True)
class Security:
    security_id: str
    issuer_id: str | None = None
    name: str = ""
    status: str = "active"


@dataclass(frozen=True)
class SymbolMapping:
    security_id: str
    symbol: str
    start: pd.Timestamp
    end: pd.Timestamp | None
    known_at: pd.Timestamp
    provider: str = ""


def _iso(value: DateLike) -> str:
    return timestamp(value).isoformat(timespec="nanoseconds")


def _interval(start, end):
    first, stop = _iso(start), None if end is None else _iso(end)
    if stop is not None and first >= stop:
        raise ValueError("start must precede exclusive end")
    return first, stop


class SecurityMaster:
    """SQLite security master; all effective intervals are half open and UTC."""

    def __init__(self, path: str | Path = ".convexity/universes.sqlite3"):
        self._defer_commit = False
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS securities (
                security_id TEXT PRIMARY KEY, issuer_id TEXT, name TEXT NOT NULL,
                status TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS symbols (
                security_id TEXT NOT NULL REFERENCES securities(security_id),
                symbol TEXT NOT NULL, start TEXT NOT NULL, end TEXT,
                known_at TEXT NOT NULL, provider TEXT NOT NULL DEFAULT '',
                UNIQUE(security_id,symbol,start,provider));
            CREATE INDEX IF NOT EXISTS symbols_lookup ON symbols(symbol,start,end);
        """)

    def _commit(self):
        if not getattr(self, "_defer_commit", False):
            self.connection.commit()

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def add_security(self, security: Security | str, **kwargs) -> Security:
        if isinstance(security, str):
            security = Security(security, **kwargs)
        if not security.security_id.strip():
            raise ValueError("security_id cannot be empty")
        old = self.connection.execute(
            "SELECT * FROM securities WHERE security_id=?", (security.security_id,)
        ).fetchone()
        if old and Security(**dict(old)) != security:
            raise UniverseError(
                f"Conflicting identity metadata for {security.security_id}"
            )
        self.connection.execute(
            "INSERT OR IGNORE INTO securities VALUES (?,?,?,?)",
            (security.security_id, security.issuer_id, security.name, security.status),
        )
        self._commit()
        return security

    def security(self, security_id: str) -> Security:
        row = self.connection.execute(
            "SELECT * FROM securities WHERE security_id=?", (security_id,)
        ).fetchone()
        if not row:
            raise UniverseError(f"Unknown security {security_id}")
        return Security(**dict(row))

    def add_symbol(
        self,
        security_id: str,
        symbol: str,
        start: DateLike,
        end: DateLike | None = None,
        *,
        known_at: DateLike,
        provider: str = "",
        allow_ambiguous: bool = False,
    ) -> None:
        first, stop = _interval(start, end)
        if not symbol or symbol != symbol.strip():
            raise ValueError("symbol must be nonempty without surrounding whitespace")
        values = (security_id, symbol, first, stop, _iso(known_at), provider)
        exact = self.connection.execute(
            "SELECT * FROM symbols WHERE security_id=? AND symbol=? AND start=? AND provider=?",
            (security_id, symbol, first, provider),
        ).fetchone()
        if exact:
            if tuple(exact) == values:
                return
            raise UniverseError(
                "Conflicting symbol interval; reconcile source records explicitly"
            )
        # Different mappings may not overlap within the same provider namespace.
        conflicts = self.connection.execute(
            """SELECT * FROM symbols WHERE provider=?
            AND (security_id=? OR symbol=?) AND (? IS NULL OR start<?)
            AND (end IS NULL OR end>?)""",
            (provider, security_id, symbol, stop, stop, first),
        ).fetchall()
        if conflicts and not allow_ambiguous:
            raise UniverseError(f"Overlapping ticker identity interval for {symbol}")
        self.connection.execute("INSERT INTO symbols VALUES (?,?,?,?,?,?)", values)
        self._commit()

    def symbol_at(
        self,
        security_id: str,
        as_of: DateLike,
        *,
        known_at: DateLike | None = None,
        provider: str = "",
        allow_reconstructed: bool = False,
    ) -> str:
        when, knowledge = _iso(as_of), _iso(as_of if known_at is None else known_at)
        if knowledge > when and not allow_reconstructed:
            raise KnowledgeUnavailableError(
                "known_at cannot exceed the decision timestamp"
            )
        rows = self.connection.execute(
            """SELECT * FROM symbols WHERE security_id=? AND provider=?
            AND start<=? AND (end IS NULL OR end>?)""",
            (security_id, provider, when, when),
        ).fetchall()
        if len(rows) != 1:
            raise UniverseError(
                f"No unique historical symbol for {security_id} at {when}"
            )
        if rows[0]["known_at"] > knowledge and not allow_reconstructed:
            raise KnowledgeUnavailableError(
                f"Symbol for {security_id} was learned after {knowledge}"
            )
        return rows[0]["symbol"]

    def resolve_symbol(
        self,
        symbol: str,
        as_of: DateLike,
        *,
        provider: str = "",
        known_at: DateLike | None = None,
        allow_reconstructed: bool = False,
    ) -> Security:
        when = _iso(as_of)
        rows = self.connection.execute(
            """SELECT security_id FROM symbols WHERE symbol=? AND provider=?
            AND start<=? AND (end IS NULL OR end>?)""",
            (symbol, provider, when, when),
        ).fetchall()
        if len(rows) != 1:
            raise UniverseError(f"No unique identity for {symbol} at {when}")
        self.symbol_at(
            rows[0][0],
            as_of,
            provider=provider,
            known_at=known_at,
            allow_reconstructed=allow_reconstructed,
        )
        return self.security(rows[0][0])

    def symbol_history(
        self,
        security_id: str,
        start: DateLike,
        end: DateLike,
        *,
        provider: str = "",
        allow_reconstructed: bool = False,
    ) -> tuple[SymbolMapping, ...]:
        first, stop = _interval(start, end)
        assert stop is not None
        rows = self.connection.execute(
            """SELECT * FROM symbols WHERE security_id=? AND provider=?
            AND start<? AND (end IS NULL OR end>?) ORDER BY start""",
            (security_id, provider, stop, first),
        ).fetchall()
        result = []
        for row in rows:
            a, b = max(first, row["start"]), min(stop, row["end"] or stop)
            if row["known_at"] > a and not allow_reconstructed:
                raise KnowledgeUnavailableError(
                    f"Symbol history of {security_id} was learned after {a}"
                )
            result.append(
                SymbolMapping(
                    security_id,
                    row["symbol"],
                    timestamp(a),
                    timestamp(b),
                    timestamp(row["known_at"]),
                    provider,
                )
            )
        return tuple(result)
