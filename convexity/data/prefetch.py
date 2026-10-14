"""Resumable daily acquisition using the existing validated SQLite bar cache."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sqlite3
from typing import Callable

from convexity.backtest.cache import CachedHistoricalProvider
from convexity.backtest.data import SessionCalendar
from convexity.backtest.history import HistoryError, HistoryRequest, _digest
from .coverage import build_coverage, enforce_coverage


def daily_provider(
    *, cache: str | Path | None = ".convexity/history.sqlite3", fallback: bool = True
):
    """Use the exact native-daily namespaces populated by bulk ingestion.

    Native daily observations must be delivered after their documented
    availability; the dynamic Research path delays them until the next open.
    """
    from convexity.backtest.alpaca_daily import AlpacaDailyHistoricalData

    if not fallback:
        return (
            AlpacaDailyHistoricalData()
            if cache is None
            else CachedHistoricalProvider(AlpacaDailyHistoricalData(), cache)
        )
    from convexity.backtest.providers import FallbackHistoricalProvider

    return FallbackHistoricalProvider(primary=AlpacaDailyHistoricalData(), cache=cache)


class PermanentHistoryError(HistoryError):
    """Provider-confirmed unavailability, with durable supporting evidence.

    Empty responses, auth failures, delisting status and timeouts alone MUST NOT
    be promoted to this error. Built-in adapters conservatively leave them retryable.
    """

    def __init__(self, reason: str, *, evidence: str):
        if not evidence.strip():
            raise ValueError(
                "Permanent unavailability requires explicit provider evidence."
            )
        super().__init__(reason)
        self.evidence = evidence


@dataclass(frozen=True)
class PrefetchConfig:
    chunk_sessions: int = 63
    concurrency: int = 4
    retries: int = 2
    backoff: float = 1.0
    primary_interval: float = 0.35
    fallback_interval: float = 1.2
    missing_policy: str = "exclude"

    def __post_init__(self):
        for name in ("chunk_sessions", "concurrency", "retries"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "retries" else 1):
                raise ValueError(f"Invalid {name}.")
        for name in ("backoff", "primary_interval", "fallback_interval"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"Invalid {name}.")
        if self.missing_policy not in ("exclude", "fail"):
            raise ValueError("missing_policy must be 'exclude' or 'fail'.")


class _PacedProvider:
    def __init__(self, provider, interval):
        self.provider = provider
        self.interval = interval
        self.lock = asyncio.Lock()
        self.last = -math.inf

    async def fetch(self, request, *, refresh=False):
        # Serializing each provider also avoids concurrent IBKR client-ID reuse.
        # Primary and fallback can still run concurrently with each other.
        async with self.lock:
            loop = asyncio.get_running_loop()
            await asyncio.sleep(max(0, self.last + self.interval - loop.time()))
            self.last = loop.time()
            return await self.provider.fetch(request, refresh=refresh)


class DailyPrefetcher:
    """Store each completed chunk before beginning the next acquisition.

    Security/session provenance and confirmed failures live alongside the
    existing cache tables, not in an alternate price store. Permanent failures
    are provider-specific; a primary failure never suppresses fallback.
    """

    def __init__(
        self,
        primary=None,
        fallback=None,
        *,
        path: str | Path = ".convexity/history.sqlite3",
        config: PrefetchConfig | None = None,
        progress: Callable[[dict], None] | None = None,
    ):
        if primary is None:
            from convexity.backtest.alpaca_daily import AlpacaDailyHistoricalData

            primary = AlpacaDailyHistoricalData()
        self.path = Path(path)
        self.config = config or PrefetchConfig()
        self.progress = progress
        self.providers = []
        for provider, interval in (
            (primary, self.config.primary_interval),
            (fallback, self.config.fallback_interval),
        ):
            if provider is None:
                continue
            cache = (
                provider
                if isinstance(provider, CachedHistoricalProvider)
                else CachedHistoricalProvider(provider, self.path)
            )
            if cache.path != self.path:
                raise ValueError(
                    "Prefetch providers must use the same history cache path."
                )
            self.providers.append((cache, _PacedProvider(cache, interval)))
        initialized = self.providers[0][0]._connect()
        initialized.close()
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS prefetch_observations (
                security_id TEXT NOT NULL, calendar TEXT NOT NULL, session TEXT NOT NULL,
                symbol TEXT NOT NULL, namespace TEXT NOT NULL, price_basis TEXT NOT NULL,
                acquired_at TEXT NOT NULL, PRIMARY KEY(security_id,calendar,session))""")
            db.execute("""CREATE TABLE IF NOT EXISTS prefetch_failures (
                security_id TEXT NOT NULL, symbol TEXT NOT NULL, calendar TEXT NOT NULL,
                session TEXT NOT NULL, namespace TEXT NOT NULL, reason TEXT NOT NULL,
                permanent INTEGER NOT NULL, evidence TEXT, attempted_at TEXT NOT NULL,
                PRIMARY KEY(security_id,symbol,calendar,session,namespace))""")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    def _observed(self, calendar: str, start, end) -> set:
        """Only intact cache rows count; deleted/corrupt rows cannot fake coverage."""
        result = set()
        namespaces = {cache.namespace: cache.price_basis for cache, _ in self.providers}
        with self._connect() as db:
            rows = db.execute(
                """SELECT o.security_id,o.session,o.symbol,h.payload,h.checksum,o.namespace,o.price_basis
                FROM prefetch_observations o JOIN history_sessions h
                ON o.namespace=h.namespace AND o.calendar=h.calendar AND o.session=h.session
                AND o.symbol=h.symbol AND o.price_basis=h.price_basis
                WHERE o.calendar=? AND h.resolution='1 day' AND o.session BETWEEN ? AND ?""",
                (calendar, str(start), str(end)),
            )
            # Stream payloads rather than materializing years of OHLCV JSON.
            checked_provenance = {}
            for sid, session, symbol, payload, checksum, namespace, basis in rows:
                if namespace not in namespaces or namespaces[namespace] not in (
                    None,
                    basis,
                ):
                    continue
                try:
                    record = json.loads(payload)
                    identity = record["provenance"]
                    if identity not in checked_provenance:
                        acquisition = db.execute(
                            "SELECT payload FROM history_acquisitions WHERE identity=?",
                            (identity,),
                        ).fetchone()
                        checked_provenance[identity] = (
                            acquisition is not None
                            and _digest(json.loads(acquisition[0])) == identity
                        )
                    if _digest(record) == checksum and checked_provenance[identity]:
                        result.add((sid, session, symbol))
                except TypeError, ValueError, KeyError:
                    pass
        return result

    def _failure(self, sid, symbol, calendar, sessions, namespace, error):
        permanent = isinstance(error, PermanentHistoryError)
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            db.executemany(
                "INSERT OR REPLACE INTO prefetch_failures VALUES (?,?,?,?,?,?,?,?,?)",
                [
                    (
                        sid,
                        symbol,
                        calendar,
                        str(s.label),
                        namespace,
                        str(error),
                        int(permanent),
                        getattr(error, "evidence", None),
                        now,
                    )
                    for s in sessions
                ],
            )

    async def run(
        self,
        members,
        start: date | str,
        end: date | str,
        *,
        calendar="XNYS",
        resume=True,
        retry_permanent=False,
        acquisition_members=None,
    ) -> dict:
        """Acquire half-open HistoricalMember intervals within inclusive dates.

        ``resume=False`` refreshes observations. Transient errors are retried on
        every invocation; permanent errors require ``retry_permanent=True``.
        """
        request = HistoryRequest(("SPY",), start, end, "1 day", calendar)
        table = SessionCalendar.from_exchange(calendar, request.start, request.end)
        members = tuple(members)
        acquisition_members = (
            members if acquisition_members is None else tuple(acquisition_members)
        )
        observed = (
            self._observed(calendar, request.start, request.end) if resume else set()
        )
        eligible = {}
        work = {}
        now = datetime.now(timezone.utc)
        attempted_since = now.isoformat()
        for member in members:
            for session in table.sessions:
                if not member.start <= session.open < member.end:
                    continue
                groups = eligible.setdefault(str(session.label), {})
                for universe in member.universes:
                    groups.setdefault(universe, set()).add(member.security_id)
                groups.setdefault("UNION", set()).add(member.security_id)
        for member in acquisition_members:
            for session in table.sessions:
                if (
                    member.start <= session.open < member.end
                    and (member.security_id, str(session.label), member.symbol)
                    not in observed
                ):
                    work.setdefault((member.security_id, member.symbol), {})[
                        session.label
                    ] = session
        if not eligible:
            raise HistoryError(
                "No historical eligible members resolved in the requested interval."
            )
        self._emit(
            dict(
                event="planned",
                securities=len({m.security_id for m in members}),
                missing_sessions=sum(len(v) for v in work.values()),
            )
        )
        semaphore = asyncio.Semaphore(self.config.concurrency)
        completed = observed
        indices = {s.label: n for n, s in enumerate(table.sessions)}

        async def acquire(sid, symbol, sessions):
            remaining = tuple(
                s for s in sessions if (sid, str(s.label), symbol) not in completed
            )
            if not remaining:
                return
            if len(remaining) != len(sessions):
                for session in remaining:
                    await acquire(sid, symbol, (session,))
                return
            for cache, paced in self.providers:
                namespace = cache.namespace
                with self._connect() as db:
                    permanent = {
                        row[0]
                        for row in db.execute(
                            "SELECT session FROM prefetch_failures WHERE security_id=? AND symbol=? AND calendar=? "
                            "AND namespace=? AND permanent=1 AND (? OR attempted_at>=?)",
                            (
                                sid,
                                symbol,
                                calendar,
                                namespace,
                                not retry_permanent,
                                attempted_since,
                            ),
                        )
                    }
                available = tuple(s for s in sessions if str(s.label) not in permanent)
                if not available:
                    continue
                # Do not span a provider-confirmed missing day in a larger request.
                if len(available) != len(sessions):
                    for session in sessions:
                        await acquire_one_provider(
                            cache, paced, sid, symbol, (session,), permanent
                        )
                    remaining = tuple(
                        s
                        for s in sessions
                        if (sid, str(s.label), symbol) not in completed
                    )
                    if not remaining:
                        return
                    continue
                if await acquire_one_provider(
                    cache, paced, sid, symbol, sessions, permanent
                ):
                    return
            # Isolate genuinely missing dates: a delisting in a chunk must not
            # hide all earlier accessible bars. Already completed halves persist.
            if len(sessions) > 1:
                middle = len(sessions) // 2
                await acquire(sid, symbol, sessions[:middle])
                await acquire(sid, symbol, sessions[middle:])

        async def acquire_one_provider(cache, paced, sid, symbol, sessions, permanent):
            if any(str(s.label) in permanent for s in sessions):
                return False
            try:
                part = HistoryRequest(
                    (symbol,), sessions[0].label, sessions[-1].label, "1 day", calendar
                )
            except ValueError as error:
                self._failure(
                    sid,
                    symbol,
                    calendar,
                    sessions,
                    cache.namespace,
                    HistoryError(
                        f"Unusable provider symbol mapping {symbol!r}: {error}"
                    ),
                )
                return False
            for attempt in range(self.config.retries + 1):
                try:
                    dataset = await paced.fetch(part, refresh=not resume)
                    # Cache.fetch has committed this validated chunk already.
                    with self._connect() as db:
                        db.executemany(
                            "INSERT OR REPLACE INTO prefetch_observations VALUES (?,?,?,?,?,?,?)",
                            [
                                (
                                    sid,
                                    calendar,
                                    str(s.label),
                                    symbol,
                                    cache.namespace,
                                    dataset.data.price_basis,
                                    datetime.now(timezone.utc).isoformat(),
                                )
                                for s in sessions
                            ],
                        )
                        db.executemany(
                            "DELETE FROM prefetch_failures WHERE security_id=? AND symbol=? AND calendar=? AND session=? AND namespace=?",
                            [
                                (sid, symbol, calendar, str(s.label), cache.namespace)
                                for s in sessions
                            ],
                        )
                    completed.update((sid, str(s.label), symbol) for s in sessions)
                    self._emit(
                        dict(
                            event="completed",
                            security_id=sid,
                            symbol=symbol,
                            start=str(part.start),
                            end=str(part.end),
                            provider=cache.namespace,
                            cache_hit=dataset.cache_hit,
                        )
                    )
                    return True
                except (HistoryError, OSError, TimeoutError) as error:
                    if (
                        isinstance(error, PermanentHistoryError)
                        or attempt == self.config.retries
                    ):
                        self._failure(
                            sid, symbol, calendar, sessions, cache.namespace, error
                        )
                        self._emit(
                            dict(
                                event="failed",
                                security_id=sid,
                                symbol=symbol,
                                start=str(part.start),
                                end=str(part.end),
                                provider=cache.namespace,
                                reason=str(error),
                                permanent=isinstance(error, PermanentHistoryError),
                            )
                        )
                        return False
                    await asyncio.sleep(self.config.backoff * 2**attempt)
            return False

        async def worker(sid, symbol, sessions):
            async with semaphore:
                # Chunk only adjacent exchange sessions; leave cached dates alone.
                run = []
                for session in sorted(sessions.values(), key=lambda s: s.label):
                    if run and (
                        len(run) >= self.config.chunk_sessions
                        or indices[session.label] != indices[run[-1].label] + 1
                    ):
                        await acquire(sid, symbol, tuple(run))
                        run = []
                    if session.close > now:
                        self._failure(
                            sid,
                            symbol,
                            calendar,
                            (session,),
                            "calendar",
                            HistoryError("Session is unfinished or in the future."),
                        )
                    else:
                        run.append(session)
                if run:
                    await acquire(sid, symbol, tuple(run))

        # TaskGroup propagates cancellation and cancels siblings on unexpected
        # programming/storage errors. Committed chunks remain resumable.
        async with asyncio.TaskGroup() as group:
            for (sid, symbol), sessions in work.items():
                group.create_task(worker(sid, symbol, sessions))
        observable = {}
        # A corrected ticker mapping must not inherit coverage from a previously
        # cached, different historical symbol for the same ID/date.
        for member in members:
            for session in table.sessions:
                label = str(session.label)
                if (
                    member.start <= session.open < member.end
                    and (member.security_id, label, member.symbol) in completed
                ):
                    observable.setdefault(label, set()).add(member.security_id)
        with self._connect() as db:
            db.row_factory = sqlite3.Row
            failures = [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM prefetch_failures WHERE calendar=? AND session BETWEEN ? AND ? ORDER BY session,security_id,namespace",
                    (calendar, str(request.start), str(request.end)),
                )
                if row["security_id"] in {m.security_id for m in members}
            ]
        report = build_coverage(eligible, observable, failures=failures)
        report["limitations"].append(
            "Split-adjusted histories retain acquisition timestamps but not a common adjustment vintage; "
            "after corporate-action revisions refresh all required history to avoid mixing revised scales."
        )
        report.update(
            start=str(request.start),
            end=str(request.end),
            resolution="1 day",
            cache=str(self.path),
            source_namespaces=[cache.namespace for cache, _ in self.providers],
            acquisition_scope="verified symbol intervals of the historical union",
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
        self._emit(dict(event="finished", complete=report["complete"]))
        return enforce_coverage(report, self.config.missing_policy)

    def _emit(self, record):
        if self.progress is not None:
            self.progress(record)
