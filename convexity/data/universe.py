"""Historical membership independent of provider price availability."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, cast

import pandas as pd

from .core import DateLike, timestamp
from .security import (
    SecurityMaster,
    UniverseError,
    KnowledgeUnavailableError,
    _iso,
    _interval,
)


@dataclass(frozen=True)
class UniverseMember:
    security_id: str
    symbol: str


@dataclass(frozen=True)
class UniverseSnapshot:
    name: str
    as_of: pd.Timestamp
    members: tuple[UniverseMember, ...]
    warnings: tuple[str, ...] = ()

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(member.symbol for member in self.members)

    @property
    def security_ids(self) -> tuple[str, ...]:
        return tuple(member.security_id for member in self.members)

    def __len__(self):
        return len(self.members)

    def __iter__(self):
        return iter(self.members)

    def _combine(self, other, intersection):
        if not isinstance(other, UniverseSnapshot):
            return NotImplemented
        if self.as_of != other.as_of:
            raise UniverseError("Combine universes only at the same decision timestamp")
        left, right = (
            {m.security_id: m for m in self},
            {m.security_id: m for m in other},
        )
        for key in left.keys() & right.keys():
            if left[key].symbol != right[key].symbol:
                raise UniverseError(f"Conflicting historical symbol for {key}")
        keys = (
            left.keys() & right.keys() if intersection else left.keys() | right.keys()
        )
        members = tuple((left | right)[k] for k in sorted(keys))
        return UniverseSnapshot(
            f"({self.name}{'&' if intersection else '|'}{other.name})",
            self.as_of,
            members,
            tuple(dict.fromkeys((*self.warnings, *other.warnings))),
        )

    def __or__(self, other):
        return self._combine(other, False)

    def __and__(self, other):
        return self._combine(other, True)


@dataclass(frozen=True)
class HistoricalMember:
    security_id: str
    symbol: str
    start: pd.Timestamp
    end: pd.Timestamp
    universes: tuple[str, ...]


class UniverseStore(SecurityMaster):
    def __init__(self, path: str | Path = ".convexity/universes.sqlite3"):
        super().__init__(path)
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS memberships (
                name TEXT NOT NULL, security_id TEXT NOT NULL REFERENCES securities(security_id),
                start TEXT NOT NULL, end TEXT, known_at TEXT NOT NULL, source TEXT NOT NULL DEFAULT '', end_known_at TEXT NOT NULL,
                UNIQUE(name,security_id,start));
            CREATE INDEX IF NOT EXISTS membership_dates ON memberships(name,start,end);
            CREATE TABLE IF NOT EXISTS universe_coverage (
                name TEXT PRIMARY KEY, start TEXT NOT NULL, end TEXT NOT NULL,
                known_at TEXT NOT NULL, reconstructed INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS universe_sources (
                source_id TEXT PRIMARY KEY, name TEXT NOT NULL, source TEXT NOT NULL,
                retrieved_at TEXT NOT NULL, sha256 TEXT NOT NULL, raw BLOB NOT NULL,
                metadata TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS universe_discrepancies (
                name TEXT NOT NULL, source_id TEXT NOT NULL, detail TEXT NOT NULL,
                UNIQUE(name,source_id,detail));
        """)
        columns = {
            row[1] for row in self.connection.execute("PRAGMA table_info(memberships)")
        }
        if "end_known_at" not in columns:
            self.connection.execute(
                "ALTER TABLE memberships ADD COLUMN end_known_at TEXT NOT NULL DEFAULT ''"
            )
            self.connection.execute("UPDATE memberships SET end_known_at=known_at")
            self.connection.commit()

    def add_membership(
        self,
        name: str,
        security_id: str,
        start: DateLike,
        end: DateLike | None = None,
        *,
        known_at: DateLike,
        source: str = "",
        end_known_at: DateLike | None = None,
    ) -> None:
        name = name.upper()
        first, stop = _interval(start, end)
        values = (
            name,
            security_id,
            first,
            stop,
            _iso(known_at),
            source,
            _iso(known_at if end_known_at is None else end_known_at),
        )
        exact = self.connection.execute(
            "SELECT * FROM memberships WHERE name=? AND security_id=? AND start=?",
            (name, security_id, first),
        ).fetchone()
        if exact:
            if tuple(exact) == values:
                return
            raise UniverseError(
                "Conflicting membership interval; reconcile source records explicitly"
            )
        conflict = self.connection.execute(
            """SELECT 1 FROM memberships WHERE name=? AND security_id=?
            AND (? IS NULL OR start<?) AND (end IS NULL OR end>?)""",
            (name, security_id, stop, stop, first),
        ).fetchone()
        if conflict:
            raise UniverseError(f"Overlapping membership for {security_id} in {name}")
        self.connection.execute(
            "INSERT INTO memberships VALUES (?,?,?,?,?,?,?)", values
        )
        self._commit()

    def set_coverage(
        self,
        name: str,
        start: DateLike,
        end: DateLike,
        *,
        known_at: DateLike,
        reconstructed: bool = False,
    ):
        first, stop = _interval(start, end)
        self.connection.execute(
            "INSERT OR REPLACE INTO universe_coverage VALUES (?,?,?,?,?)",
            (name.upper(), first, stop, _iso(known_at), int(reconstructed)),
        )
        self._commit()

    def _coverage(self, name, first, stop, knowledge, allow_reconstructed):
        row = self.connection.execute(
            "SELECT * FROM universe_coverage WHERE name=?", (name,)
        ).fetchone()
        if row:
            if first < row["start"] or stop > row["end"]:
                raise UniverseError(
                    f"{name} reference coverage is [{row['start']}, {row['end']}); requested interval is outside it"
                )
            if not allow_reconstructed and (
                row["reconstructed"] or row["known_at"] > knowledge
            ):
                raise KnowledgeUnavailableError(
                    f"{name} is retrospective or was learned after the decision; explicitly set allow_reconstructed=True for research"
                )
        elif not self.connection.execute(
            "SELECT 1 FROM memberships WHERE name=?", (name,)
        ).fetchone():
            raise UniverseError(
                f"No membership data for {name}; import a historical source first"
            )

    def discrepancies(self, name: str) -> tuple[str, ...]:
        return tuple(
            row[0]
            for row in self.connection.execute(
                "SELECT DISTINCT detail FROM universe_discrepancies WHERE name=? ORDER BY detail",
                (name.upper(),),
            )
        )

    def sources(self, name: str) -> tuple[dict, ...]:
        return tuple(
            {
                **{
                    k: row[k] for k in ("source_id", "source", "retrieved_at", "sha256")
                },
                "metadata": json.loads(row["metadata"]),
            }
            for row in self.connection.execute(
                "SELECT * FROM universe_sources WHERE name=?", (name.upper(),)
            )
        )

    def resolve(
        self,
        name: str,
        as_of: DateLike,
        *,
        known_at: DateLike | None = None,
        allow_reconstructed: bool = False,
    ) -> UniverseSnapshot:
        name, when = name.upper(), timestamp(as_of)
        knowledge = timestamp(as_of if known_at is None else known_at)
        if knowledge > when and not allow_reconstructed:
            raise KnowledgeUnavailableError(
                "known_at cannot exceed the decision timestamp"
            )
        self._coverage(
            name,
            _iso(when),
            _iso(cast(pd.Timestamp, when + pd.Timedelta(1, "ns"))),
            _iso(knowledge),
            allow_reconstructed,
        )
        rows = self.connection.execute(
            """SELECT * FROM memberships WHERE name=? AND start<=?
            AND (end IS NULL OR end>?) ORDER BY security_id""",
            (name, _iso(when), _iso(when)),
        ).fetchall()
        if not allow_reconstructed:
            hidden_removal = self.connection.execute(
                """SELECT security_id FROM memberships
                WHERE name=? AND end<=? AND end_known_at>?""",
                (name, _iso(when), _iso(knowledge)),
            ).fetchone()
            if hidden_removal:
                raise KnowledgeUnavailableError(
                    f"Removal of {hidden_removal[0]} was learned after the decision timestamp"
                )
        members = []
        for row in rows:
            if row["known_at"] > _iso(knowledge) and not allow_reconstructed:
                raise KnowledgeUnavailableError(
                    f"Membership of {row['security_id']} was learned after {knowledge}"
                )
            symbol = self.symbol_at(
                row["security_id"],
                when,
                known_at=knowledge,
                allow_reconstructed=allow_reconstructed,
            )
            members.append(UniverseMember(row["security_id"], symbol))
        warnings = self.discrepancies(name)
        if allow_reconstructed:
            warnings += (
                "Retrospective reconstruction allowed; publication-time correctness is not established.",
            )
        return UniverseSnapshot(name, when, tuple(members), warnings)

    def historical_members(
        self,
        names: str | Iterable[str],
        start: DateLike,
        end: DateLike,
        *,
        allow_reconstructed: bool = False,
    ) -> tuple[HistoricalMember, ...]:
        first, stop = _interval(start, end)
        assert stop is not None
        names = [names] if isinstance(names, str) else list(names)
        pieces = {}
        for name in dict.fromkeys(n.upper() for n in names):
            self._coverage(name, first, stop, first, allow_reconstructed)
            rows = self.connection.execute(
                """SELECT m.security_id,m.start AS member_start,m.end AS member_end,
                m.known_at AS membership_known,s.symbol,s.start,s.end,s.known_at
                FROM memberships m JOIN symbols s ON m.security_id=s.security_id AND s.provider=''
                WHERE m.name=? AND m.start<? AND (m.end IS NULL OR m.end>?)
                  AND s.start<? AND (s.end IS NULL OR s.end>?)""",
                (name, stop, first, stop, first),
            ).fetchall()
            expected = self.connection.execute(
                """SELECT * FROM memberships WHERE name=? AND start<?
                AND (end IS NULL OR end>?)""",
                (name, stop, first),
            ).fetchall()
            if not allow_reconstructed:
                for member in expected:
                    if (
                        member["end"] is not None
                        and member["end"] < stop
                        and member["end_known_at"] > member["end"]
                    ):
                        raise KnowledgeUnavailableError(
                            f"Removal of {member['security_id']} was learned after its effective date"
                        )
            covered = {}
            for row in rows:
                a = max(first, row["start"], row["member_start"])
                b = min(stop, row["end"] or stop, row["member_end"] or stop)
                if a >= b:
                    continue
                if (
                    not allow_reconstructed
                    and max(row["membership_known"], row["known_at"]) > a
                ):
                    raise KnowledgeUnavailableError(
                        f"Historical reference data for {row['security_id']} was learned after its effective interval"
                    )
                pieces.setdefault((row["security_id"], row["symbol"], a, b), set()).add(
                    name
                )
                covered.setdefault(row["security_id"], []).append((a, b))
            for row in expected:
                cursor, finish = max(first, row["start"]), min(stop, row["end"] or stop)
                for a, b in sorted(covered.get(row["security_id"], [])):
                    if a > cursor:
                        break
                    cursor = max(cursor, b)
                if cursor < finish:
                    raise UniverseError(
                        f"Missing historical symbol interval for {row['security_id']} at {cursor}"
                    )
        # Split overlaps across indexes into one canonical set of identity/symbol
        # segments. A security joining a second index must not be downloaded twice.
        grouped = {}
        for (sid, symbol, a, b), universes in pieces.items():
            grouped.setdefault((sid, symbol), []).append((a, b, universes))
        result = []
        for (sid, symbol), intervals in sorted(grouped.items()):
            boundaries = sorted({point for a, b, _ in intervals for point in (a, b)})
            for a, b in zip(boundaries, boundaries[1:]):
                names_at_time = tuple(
                    sorted(
                        set().union(*(names for x, y, names in intervals if x <= a < y))
                    )
                )
                if not names_at_time:
                    continue
                if (
                    result
                    and result[-1].security_id == sid
                    and result[-1].symbol == symbol
                    and result[-1].end == timestamp(a)
                    and result[-1].universes == names_at_time
                ):
                    previous = result.pop()
                    result.append(
                        HistoricalMember(
                            sid, symbol, previous.start, timestamp(b), names_at_time
                        )
                    )
                else:
                    result.append(
                        HistoricalMember(
                            sid, symbol, timestamp(a), timestamp(b), names_at_time
                        )
                    )
        return tuple(result)


def universe(
    name: str,
    as_of: DateLike,
    *,
    store: UniverseStore | str | Path | None = None,
    known_at: DateLike | None = None,
    allow_reconstructed: bool = False,
) -> UniverseSnapshot:
    """Resolve a historical eligible set, retaining members without price data."""
    if isinstance(store, UniverseStore):
        return store.resolve(
            name, as_of, known_at=known_at, allow_reconstructed=allow_reconstructed
        )
    with UniverseStore(store or ".convexity/universes.sqlite3") as opened:
        return opened.resolve(
            name, as_of, known_at=known_at, allow_reconstructed=allow_reconstructed
        )
