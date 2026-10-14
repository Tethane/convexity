"""Persistent, validated historical observations shared across research runs."""

from __future__ import annotations

from bisect import bisect_right
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Protocol

from convexity.runtime.types import Bar
from .bar_size import parse_bar_size
from .data import HistoricalData, SessionCalendar
from .history import (
    HistoricalDataset,
    HistoryError,
    HistoryRequest,
    _digest,
    validate_coverage,
)


class _Provider(Protocol):
    async def fetch(
        self, request: HistoryRequest, *, refresh: bool = False
    ) -> HistoricalDataset: ...


class CachedHistoricalProvider:
    """Cache complete symbol/session panels in SQLite, fetching only missing runs.

    The namespace includes provider class, source, calendar, resolution and price
    basis. Providers with different feeds/settings should expose a stable
    ``source`` or ``cache_identity`` attribute, or pass ``source=`` explicitly.
    Multi-session observations are built from cached daily observations so that
    changing request boundaries cannot reuse incorrectly aligned aggregates.
    Corrupt observations fail closed; ``refresh=True`` explicitly replaces the
    requested coverage. Refresh does not delete unrelated instruments or dates.
    """

    def __init__(
        self,
        provider: _Provider,
        path: str | Path = ".convexity/history.sqlite3",
        *,
        source: str | None = None,
        price_basis: str | None = None,
    ):
        self.provider = provider
        self.path = Path(path)
        cls = type(provider)
        identity = (
            source
            or getattr(provider, "cache_identity", None)
            or getattr(provider, "source", None)
            or "default"
        )
        self.namespace = f"{cls.__module__}.{cls.__qualname__}:{identity}"
        self.price_basis = price_basis or getattr(provider, "price_basis", None)

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute("PRAGMA journal_mode=WAL")
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise HistoryError(f"Unsupported historical cache schema {version}.")
            db.execute("""CREATE TABLE IF NOT EXISTS history_sessions (
                namespace TEXT NOT NULL, calendar TEXT NOT NULL, resolution TEXT NOT NULL,
                price_basis TEXT NOT NULL, symbol TEXT NOT NULL, session TEXT NOT NULL,
                payload TEXT NOT NULL, checksum TEXT NOT NULL,
                PRIMARY KEY (namespace, calendar, resolution, price_basis, symbol, session))""")
            db.execute("PRAGMA user_version=1")
            db.execute(
                "CREATE TABLE IF NOT EXISTS history_acquisitions (identity TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
            db.commit()
            return db
        except BaseException:
            db.close()
            raise

    async def fetch(
        self, request: HistoryRequest, *, refresh: bool = False
    ) -> HistoricalDataset:
        if not isinstance(request, HistoryRequest):
            raise TypeError("fetch requires HistoryRequest.")
        calendar = SessionCalendar.from_exchange(
            request.calendar, request.start, request.end
        )
        if not calendar.sessions:
            raise HistoryError("The requested period contains no exchange sessions.")
        if calendar.sessions[-1].close > datetime.now(timezone.utc):
            raise HistoryError(
                "The requested period includes an unfinished or future session."
            )
        resolution = (
            "1 day" if parse_bar_size(request.bar_size).multiday else request.bar_size
        )
        base = HistoryRequest(
            request.symbols, request.start, request.end, resolution, request.calendar
        )
        key = (self.namespace, request.calendar, resolution)
        records = {}
        decoded_bars = {}
        basis = self.price_basis
        db = None
        try:
            db = self._connect()
            # Bulk acquisition issues narrow symbol requests against a large
            # shared store. Do not load every other constituent's payload for
            # each chunk. Bound placeholders for older SQLite variable limits.
            filter_symbols = len(request.symbols) <= 500
            symbol_filter = (
                " AND symbol IN (" + ",".join("?" for _ in request.symbols) + ")"
                if filter_symbols
                else ""
            )
            basis_filter = " AND price_basis=?" if basis is not None else ""
            rows = (
                []
                if refresh
                else db.execute(
                    "SELECT price_basis, symbol, session, payload, checksum FROM history_sessions "
                    "WHERE namespace=? AND calendar=? AND resolution=? AND session BETWEEN ? AND ?"
                    + symbol_filter
                    + basis_filter,
                    (
                        *key,
                        str(request.start),
                        str(request.end),
                        *(request.symbols if filter_symbols else ()),
                        *((basis,) if basis is not None else ()),
                    ),
                ).fetchall()
            )
            relevant = [
                row
                for row in rows
                if row[1] in request.symbols and (basis is None or row[0] == basis)
            ]
            bases = {row[0] for row in relevant}
            if len(bases) > 1:
                raise HistoryError(
                    "Multiple cached price bases; specify price_basis explicitly."
                )
            if bases:
                basis = next(iter(bases))
            sessions = {str(s.label): s for s in calendar.sessions}
            for cached_basis, symbol, label, payload, checksum in relevant:
                if label not in sessions:
                    raise ValueError("Cached session is absent from exchange calendar.")
                record = json.loads(payload)
                if _digest(record) != checksum:
                    raise ValueError("Cached observation checksum mismatch.")
                session = sessions[label]
                if (
                    record["open"] != session.open.isoformat()
                    or record["close"] != session.close.isoformat()
                ):
                    raise ValueError("Cached session differs from exchange calendar.")
                single = HistoricalData(
                    tuple(self._bar(row) for row in record["bars"]),
                    record["source"],
                    cached_basis,
                )
                validate_coverage(
                    single,
                    SessionCalendar((session,), calendar.name),
                    HistoryRequest(
                        (symbol,),
                        session.label,
                        session.label,
                        resolution,
                        request.calendar,
                    ),
                )
                records[symbol, label] = record
                decoded_bars[symbol, label] = single.bars

            # A missing run is contiguous in the exchange calendar, not civil
            # dates: holidays/weekends never cause unnecessary acquisitions.
            missing = {}
            for symbol in request.symbols:
                run = []
                for session in calendar.sessions:
                    if (symbol, str(session.label)) not in records:
                        run.append(session)
                    elif run:
                        missing.setdefault((run[0].label, run[-1].label), []).append(
                            symbol
                        )
                        run = []
                if run:
                    missing.setdefault((run[0].label, run[-1].label), []).append(symbol)
            pending = []
            provenance = {}
            for (start, end), symbols in sorted(missing.items()):
                part = HistoryRequest(
                    tuple(symbols), start, end, resolution, request.calendar
                )
                dataset = await self.provider.fetch(part, refresh=refresh)
                authoritative = SessionCalendar.from_exchange(
                    request.calendar, start, end
                )
                if dataset.calendar.sessions != authoritative.sessions:
                    raise HistoryError(
                        "Provider returned an incomplete or different exchange calendar."
                    )
                validate_coverage(dataset.data, authoritative, part)
                if basis is not None and dataset.data.price_basis != basis:
                    raise HistoryError(
                        "Provider price basis differs from cached/requested history."
                    )
                basis = dataset.data.price_basis
                provenance_id = _digest(dataset.manifest)
                provenance[provenance_id] = dataset.manifest
                if records and any(
                    record["source"] != dataset.data.source
                    for record in records.values()
                ):
                    raise HistoryError(
                        "Provider source differs from cached history; use a distinct cache namespace."
                    )
                # Partition once, instead of scanning the complete dataset for
                # every symbol/session (quadratic on multi-year minute history).
                opens = [session.open for session in authoritative.sessions]
                grouped = {}
                for bar in dataset.data.bars:
                    session = authoritative.sessions[bisect_right(opens, bar.start) - 1]
                    grouped.setdefault((bar.symbol, str(session.label)), []).append(bar)
                for session in authoritative.sessions:
                    for symbol in symbols:
                        bars = grouped[symbol, str(session.label)]
                        record = dict(
                            open=session.open.isoformat(),
                            close=session.close.isoformat(),
                            source=dataset.data.source,
                            bars=[
                                dict(
                                    symbol=b.symbol,
                                    start=b.start.isoformat(),
                                    end=b.end.isoformat(),
                                    open=str(b.open),
                                    high=str(b.high),
                                    low=str(b.low),
                                    close=str(b.close),
                                    volume=b.volume,
                                )
                                for b in bars
                            ],
                            provenance=provenance_id,
                        )
                        records[symbol, str(session.label)] = record
                        decoded_bars[symbol, str(session.label)] = tuple(bars)
                        pending.append(
                            (
                                *key,
                                basis,
                                symbol,
                                str(session.label),
                                json.dumps(
                                    record,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                    allow_nan=False,
                                ),
                                _digest(record),
                            )
                        )
            source_names = {record["source"] for record in records.values()}
            if len(source_names) != 1 or basis is None:
                raise HistoryError(
                    "Historical cache contains inconsistent source metadata."
                )
            data = HistoricalData(
                tuple(bar for bars in decoded_bars.values() for bar in bars),
                next(iter(source_names)),
                basis,
            )
            validate_coverage(data, calendar, base)
            if resolution != request.bar_size:
                data = data.resample(request.bar_size, calendar)
                validate_coverage(data, calendar, request)
            for record in records.values():
                identity = record["provenance"]
                if identity not in provenance:
                    row = db.execute(
                        "SELECT payload FROM history_acquisitions WHERE identity=?",
                        (identity,),
                    ).fetchone()
                    if row is None:
                        raise ValueError("Missing acquisition provenance.")
                    payload = json.loads(row[0])
                    if _digest(payload) != identity:
                        raise ValueError("Acquisition provenance checksum mismatch.")
                    provenance[identity] = payload
            # Commit only after the entire request has passed validation.
            with db:
                db.executemany(
                    "INSERT OR REPLACE INTO history_sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    pending,
                )
                db.executemany(
                    "INSERT OR REPLACE INTO history_acquisitions VALUES (?, ?)",
                    [
                        (
                            identity,
                            json.dumps(
                                payload,
                                sort_keys=True,
                                separators=(",", ":"),
                                allow_nan=False,
                            ),
                        )
                        for identity, payload in provenance.items()
                    ],
                )
            manifest = dict(
                provider=self.namespace,
                source=data.source,
                price_basis=basis,
                request=request.to_dict(),
                data_identity=data.identity,
                acquisitions=[provenance[key] for key in sorted(provenance)],
            )
            return HistoricalDataset(
                data, calendar, manifest, self.path, not bool(missing)
            )
        except (sqlite3.Error, OSError, KeyError, TypeError, ValueError) as exc:
            raise HistoryError(
                f"Historical SQLite cache is invalid or unavailable: {self.path}. "
                "Use refresh=True to reacquire corrupt observations; restore or remove a damaged database."
            ) from exc
        finally:
            if db is not None:
                db.close()

    @staticmethod
    def _bar(row):
        return Bar(
            row["symbol"],
            datetime.fromisoformat(row["start"]),
            datetime.fromisoformat(row["end"]),
            row["open"],
            row["high"],
            row["low"],
            row["close"],
            row["volume"],
        )
