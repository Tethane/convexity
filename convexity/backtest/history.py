"""Real IBKR stock history, complete exchange sessions, and immutable local snapshots.

TRADES prices are split-adjusted, not dividend-adjusted. These datasets support
price-return research in adjusted share units; they do not model dividend cash
flows or point-in-time corporate-action/universe membership information.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Callable, cast
from uuid import uuid4

from convexity._ibkr_connection import connect_readonly
from convexity.brokers.ibkr import PaperAccountConfig, _connect_paper, _new_client
from convexity.runtime.types import Bar
from .data import HistoricalData, Session, SessionCalendar
from .bar_size import intervals, parse_bar_size

PRICE_BASIS = "split_adjusted_price_return"
SCHEMA = 1


class HistoryError(RuntimeError):
    """Real historical data is unavailable, incomplete, or unverifiable."""


@dataclass(frozen=True)
class HistoricalConnectionConfig:
    """Read-only connection; None host/port enables WSL and Gateway/TWS discovery."""

    host: str | None = None
    port: int | None = None
    client_id: int = 72
    timeout: float = 15.0

    def __post_init__(self):
        if self.host is not None and (not isinstance(self.host, str) or not self.host.strip()):
            raise ValueError("host must be a nonempty hostname or IP address.")
        if self.port is not None and (type(self.port) is not int or not 1 <= self.port <= 65535):
            raise ValueError("port must be an integer between 1 and 65535.")
        if type(self.client_id) is not int or not 1 <= self.client_id <= 2**31 - 1:
            raise ValueError("client_id must be a positive 32-bit integer.")
        if type(self.timeout) not in (int, float) or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be finite and positive.")

    @classmethod
    def from_env(cls) -> HistoricalConnectionConfig:
        return cls(host=os.environ.get("IBKR_HOST"),
                   port=int(os.environ["IBKR_PORT"]) if "IBKR_PORT" in os.environ else None,
                   client_id=int(os.environ.get("IBKR_CLIENT_ID", "72")),
                   timeout=float(os.environ.get("IBKR_TIMEOUT", "15")))


@dataclass(frozen=True)
class HistoryRequest:
    """Inclusive exchange-session dates; only complete regular sessions are used."""
    symbols: tuple[str, ...]
    start: date | str
    end: date | str
    bar_size: str = "1 day"
    calendar: str = "XNYS"

    def __post_init__(self):
        if isinstance(self.symbols, str):
            raise ValueError("symbols must be a sequence, for example ('SPY',).")
        if not self.symbols or any(not isinstance(s, str) or not re.fullmatch(r"[A-Z0-9][A-Z0-9 .-]*", s) for s in self.symbols):
            raise ValueError("Supply nonempty uppercase stock symbols.")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("Stock symbols must be unique.")
        object.__setattr__(self, "symbols", tuple(sorted(self.symbols)))
        for key in ("start", "end"):
            value = getattr(self, key)
            value = date.fromisoformat(value) if isinstance(value, str) else value
            if type(value) is not date:
                raise ValueError("start/end must be ISO dates or date objects, not datetimes.")
            object.__setattr__(self, key, value)
        if cast(date, self.start) > cast(date, self.end):
            raise ValueError("start must not follow end.")
        object.__setattr__(self, "bar_size", parse_bar_size(self.bar_size).label)
        if self.calendar not in ("XNYS", "XNAS", "NASDAQ"):
            raise ValueError("IBKR research currently supports US USD stocks on XNYS/XNAS/NASDAQ calendars.")

    def to_dict(self) -> dict:
        return dict(symbols=list(self.symbols), start=str(self.start), end=str(self.end),
                    bar_size=self.bar_size, calendar=self.calendar)


@dataclass(frozen=True)
class HistoricalDataset:
    data: HistoricalData
    calendar: SessionCalendar
    manifest: dict
    cache_path: Path
    cache_hit: bool = False


def _encoded(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_encoded(value)).hexdigest()


def _atomic(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_bytes(_encoded(value) + b"\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _expected(sessions: tuple[Session, ...], bar_size: str) -> set[tuple[datetime, datetime]]:
    return set(intervals(sessions, bar_size))


def _aggregate(bars: list[Bar], sessions: tuple[Session, ...], bar_size: str) -> list[Bar]:
    """Aggregate already validated source observations into requested OHLCV."""
    ordered = sorted(bars, key=lambda bar: bar.start)
    result = []
    offset = 0
    closed_gaps = {(left.close, right.open) for left, right in zip(sessions, sessions[1:])}
    for start, end in sorted(_expected(sessions, bar_size)):
        stop = offset
        while stop < len(ordered) and ordered[stop].start < end:
            stop += 1
        group = ordered[offset:stop]
        if (not group or group[0].start != start or group[-1].end != end
                or any(left.end != right.start and (left.end, right.start) not in closed_gaps
                       for left, right in zip(group, group[1:]))):
            raise HistoryError(f"Incomplete source coverage for aggregation at {start}.")
        result.append(Bar(group[0].symbol, start, end, group[0].open,
                          max(bar.high for bar in group), min(bar.low for bar in group),
                          group[-1].close, sum(bar.volume for bar in group)))
        offset = stop
    if offset != len(ordered):
        raise HistoryError("Source bars extend beyond the aggregation calendar.")
    return result


def _chunks(sessions: tuple[Session, ...], source: str):
    if source in ("1 day", "1 min"):
        # Keep responses to roughly 2,000 source bars, and durations within
        # IBKR's 365-day maximum for minute-or-larger bars. Preserve every
        # source minute: coarser native bars cannot prove complete coverage.
        offset = 0
        while offset < len(sessions):
            stop, count = offset, 0
            while stop < len(sessions):
                session = sessions[stop]
                rows = 1 if source == "1 day" else int((session.close - session.open).total_seconds() / 60)
                days = (session.close.date() - sessions[offset].open.date()).days + 1
                if stop > offset and (count + rows > 2000 or days > 365):
                    break
                count += rows
                stop += 1
            group: tuple[Session, ...] = sessions[offset:stop]
            assert group
            days = (group[-1].close.date() - group[0].open.date()).days + 1
            yield group, f"{days} D"
            offset = stop

    else:
        # IBKR step-size limits: 1s/1800s, 5s/3600s, 10-15s/14400s, 30s/28800s.
        seconds = parse_bar_size(source).count
        limit = 1800 if seconds == 1 else 3600 if seconds == 5 else 14400 if seconds < 30 else 28800
        for session in sessions:
            start = session.open
            while start < session.close:
                end = min(start + timedelta(seconds=limit), session.close)
                yield (Session(session.label, start, end),), f"{int((end - start).total_seconds())} S"
                start = end


def validate_coverage(data: HistoricalData, calendar: SessionCalendar, request: HistoryRequest):
    """Reject missing sessions/panels, extra instruments, and invented intervals."""
    if not calendar.sessions:
        raise HistoryError("The requested period contains no exchange sessions.")
    if any(not cast(date, request.start) <= s.label <= cast(date, request.end) for s in calendar.sessions):
        raise HistoryError("Cached calendar falls outside the requested dates.")
    expected = _expected(calendar.sessions, request.bar_size)
    if {b.symbol for b in data.bars} != set(request.symbols):
        raise HistoryError("Historical symbols differ from the requested universe.")
    for symbol in request.symbols:
        actual = {(b.start, b.end) for b in data.bars if b.symbol == symbol}
        missing, extra = expected - actual, actual - expected
        if missing or extra:
            example = min(missing or extra)[0].isoformat()
            raise HistoryError(f"Incomplete historical coverage for {symbol}: {len(missing)} missing, {len(extra)} unexpected bars; first at {example}. No prices were filled or fabricated.")


class IBKRHistoricalData:
    """Read-only acquisition with exact-request cache reuse and explicit refresh.

    Intraday requests acquire minute or second bars and aggregate from the session
    open. Longer periods aggregate daily observations. Daily requests cover at most 120
    sessions each. Every chunk must be complete before a snapshot is committed.
    Requests run sequentially with configurable spacing. IBKR may still throttle
    or deny data based on entitlements/availability; those errors are surfaced.
    """
    source = "IBKR:TRADES:historical"
    price_basis = PRICE_BASIS

    def __init__(self, config: PaperAccountConfig | HistoricalConnectionConfig | None = None, *, cache_dir: str | Path = ".convexity/history",
                 request_interval: float = 0.35, request_timeout: float = 60,
                 client_factory: Callable[[], Any] | None = None):
        for name, value, zero in (("request_interval", request_interval, True), ("request_timeout", request_timeout, False)):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (not zero and value == 0):
                raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}.")
        self.config = config if config is not None else HistoricalConnectionConfig.from_env()
        self.cache_dir = Path(cache_dir)
        self.request_interval = request_interval
        self.request_timeout = request_timeout
        self.client_factory = client_factory or _new_client

    def _index(self, request: HistoryRequest) -> Path:
        return self.cache_dir / "requests" / (_digest(dict(schema=SCHEMA, request=request.to_dict(), price_basis=PRICE_BASIS)) + ".json")

    def _load(self, request: HistoryRequest) -> HistoricalDataset | None:
        index = self._index(request)
        if not index.exists():
            return None
        try:
            reference = json.loads(index.read_text())
            identity = reference["sha256"]
            if not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{64}", identity):
                raise ValueError("Invalid snapshot identity.")
            path = self.cache_dir / "snapshots" / (identity + ".json")
            payload = json.loads(path.read_text())
            if _digest(payload) != identity or payload["schema"] != SCHEMA or payload["request"] != request.to_dict():
                raise ValueError("Snapshot hash/schema/request mismatch.")
            manifest = payload["manifest"]
            if manifest["provider"] != "IBKR" or manifest["price_basis"] != PRICE_BASIS:
                raise ValueError("Snapshot provider or price basis mismatch.")
            calendar = SessionCalendar(tuple(Session(date.fromisoformat(s["label"]), datetime.fromisoformat(s["open"]),
                                                     datetime.fromisoformat(s["close"])) for s in payload["sessions"]), payload["calendar_name"])
            data = HistoricalData(tuple(Bar(r["symbol"], datetime.fromisoformat(r["start"]), datetime.fromisoformat(r["end"]),
                                            r["open"], r["high"], r["low"], r["close"], r["volume"]) for r in payload["bars"]),
                                  "IBKR:TRADES:historical", PRICE_BASIS)
            if data.identity != manifest["data_identity"]:
                raise ValueError("Dataset identity mismatch.")
            authoritative = SessionCalendar.from_exchange(request.calendar, request.start, request.end)
            if calendar.sessions != authoritative.sessions:
                raise HistoryError("Cached sessions differ from the requested exchange calendar.")
            if calendar.sessions and calendar.sessions[-1].close > datetime.now(timezone.utc):
                raise HistoryError("Cached history includes an unfinished or future session.")
            validate_coverage(data, calendar, request)
            return HistoricalDataset(data, calendar, dict(manifest, snapshot_sha256=identity), path, True)
        except (OSError, KeyError, TypeError, ValueError, HistoryError) as exc:
            raise HistoryError(f"Historical cache is invalid: {index}. Use refresh=True to explicitly reacquire it.") from exc

    async def fetch(self, request: HistoryRequest, *, refresh: bool = False) -> HistoricalDataset:
        if not isinstance(request, HistoryRequest):
            raise TypeError("fetch requires HistoryRequest.")
        if not refresh:
            cached = self._load(request)
            if cached is not None:
                return cached
        calendar = SessionCalendar.from_exchange(request.calendar, request.start, request.end)
        if not calendar.sessions:
            raise HistoryError("The requested period contains no exchange sessions.")
        if calendar.sessions[-1].close > datetime.now(timezone.utc):
            raise HistoryError("The requested period includes an unfinished or future session; select completed historical dates.")
        started_at = datetime.now(timezone.utc)
        client = self.client_factory()
        chunks = []
        contracts = {}
        bars: list[Bar] = []
        raw_seen: dict[tuple[str, str], dict] = {}
        spec = parse_bar_size(request.bar_size)
        source_bar_size = spec.source
        try:
            if isinstance(self.config, PaperAccountConfig):
                async with asyncio.timeout(self.config.timeout):
                    await _connect_paper(client, self.config)
            else:
                await connect_readonly(client, host=self.config.host, port=self.config.port,
                                       client_id=self.config.client_id, timeout=self.config.timeout)
            from ib_async import Stock
            for symbol in request.symbols:
                async with asyncio.timeout(self.request_timeout):
                    details = await client.reqContractDetailsAsync(Stock(symbol, "SMART", "USD"))
                if len(details) != 1:
                    raise HistoryError(f"{symbol} did not resolve to one stock contract.")
                contract = details[0].contract
                if contract.secType != "STK" or contract.currency != "USD" or contract.symbol != symbol or contract.conId <= 0:
                    raise HistoryError(f"Resolved contract differs from requested USD stock {symbol}.")
                contracts[symbol] = {key: getattr(contract, key, "") for key in
                                     ("conId", "symbol", "secType", "currency", "exchange", "primaryExchange", "localSymbol", "tradingClass")}
                symbol_bars: list[Bar] = []
                for sessions, duration in _chunks(calendar.sessions, source_bar_size):
                    if chunks:
                        # Small-bar requests are limited to 60 per ten minutes.
                        spacing = max(self.request_interval, 10.1) if source_bar_size.endswith("secs") else self.request_interval
                        await asyncio.sleep(spacing)
                    end = sessions[-1].close
                    kwargs = dict(endDateTime=end, durationStr=duration, barSizeSetting=source_bar_size,
                                  whatToShow="TRADES", useRTH=True, formatDate=2, keepUpToDate=False,
                                  timeout=self.request_timeout)
                    async with asyncio.timeout(self.request_timeout + 1):
                        raw = await client.reqHistoricalDataAsync(contract, **kwargs)
                    if not client.isConnected():
                        raise HistoryError("IBKR disconnected during historical acquisition.")
                    converted, source_rows = self._convert(symbol, raw, sessions, source_bar_size)
                    for source_row in source_rows:
                        raw_key = (symbol, source_row["date"])
                        if raw_key in raw_seen and raw_seen[raw_key] != source_row:
                            raise HistoryError(f"IBKR revised overlapping history for {symbol} at {source_row['date']} during acquisition; retry for a consistent snapshot.")
                        raw_seen[raw_key] = source_row
                    if {(b.start, b.end) for b in converted} != _expected(sessions, source_bar_size):
                        missing = _expected(sessions, source_bar_size) - {(b.start, b.end) for b in converted}
                        raise HistoryError(f"IBKR returned incomplete {source_bar_size} history for {symbol}, {sessions[0].label} through {sessions[-1].label}: {len(missing)} missing bars. Check entitlement, listing history, trading halts, and data availability. No synthetic fallback is permitted.")
                    symbol_bars.extend(converted)
                    chunks.append(dict(symbol=symbol, contract_id=contract.conId, end=end.isoformat(), duration=kwargs["durationStr"],
                                       bar_size=source_bar_size,
                                       received_at=datetime.now(timezone.utc).isoformat(), raw_count=len(source_rows),
                                       raw_sha256=_digest(source_rows), raw_rows=source_rows))
                bars.extend(_aggregate(symbol_bars, calendar.sessions, request.bar_size))
            data = HistoricalData(tuple(bars), "IBKR:TRADES:historical", PRICE_BASIS)
            validate_coverage(data, calendar, request)
        except HistoryError:
            raise
        except Exception as exc:
            raise HistoryError(f"IBKR historical acquisition failed ({type(exc).__name__}: {exc}). No backtest data was fabricated; check gateway, client ID, entitlements and the requested dates.") from exc
        finally:
            client.disconnect()
        manifest = dict(provider="IBKR", request=request.to_dict(), price_basis=PRICE_BASIS, what_to_show="TRADES", use_rth=True,
                        source_bar_size=source_bar_size,
                        aggregation="none" if request.bar_size == parse_bar_size(source_bar_size).label else
                                    "session_open_ohlcv" if spec.unit == "second" else "session_calendar_ohlcv",
                        acquired_start=started_at.isoformat(), acquired_end=datetime.now(timezone.utc).isoformat(),
                        contracts=contracts, requests=chunks, data_identity=data.identity,
                        limitations=["Split-adjusted price returns; no dividend cash flows or total-return accounting.",
                                     "Historical adjusted share units are not point-in-time executable share prices.",
                                     "Caller selects a fixed universe; delisted names and survivorship bias are not corrected.",
                                     "IBKR filters some trade types and may revise historical data; cached snapshots are immutable."])
        payload = dict(schema=SCHEMA, request=request.to_dict(), manifest=manifest, calendar_name=calendar.name,
                       sessions=[dict(label=str(s.label), open=s.open.isoformat(), close=s.close.isoformat()) for s in calendar.sessions],
                       bars=[dict(symbol=b.symbol, start=b.start.isoformat(), end=b.end.isoformat(), open=str(b.open), high=str(b.high),
                                  low=str(b.low), close=str(b.close), volume=b.volume) for b in data.bars])
        identity = _digest(payload)
        path = self.cache_dir / "snapshots" / (identity + ".json")
        _atomic(path, payload)
        _atomic(self._index(request), dict(sha256=identity))
        return HistoricalDataset(data, calendar, dict(manifest, snapshot_sha256=identity), path)

    @staticmethod
    def _convert(symbol: str, values: Any, sessions: tuple[Session, ...], bar_size: str) -> tuple[list[Bar], list[dict]]:
        by_label = {s.label: s for s in sessions}
        expected = _expected(sessions, bar_size)
        source_seconds = parse_bar_size(bar_size).count if bar_size != "1 day" else 0
        seen: dict[tuple[datetime, datetime], Bar] = {}
        source_rows = []
        for value in values:
            source_rows.append(dict(date=str(value.date), open=str(value.open), high=str(value.high), low=str(value.low),
                                    close=str(value.close), volume=str(value.volume)))
            if bar_size == "1 day":
                label = value.date
                if isinstance(label, str):
                    label = datetime.strptime(label, "%Y%m%d").date() if re.fullmatch(r"\d{8}", label) else date.fromisoformat(label)
                if type(label) is not date:
                    raise HistoryError("IBKR daily bars must identify a session date.")
                session = by_label.get(label)
                if session is None:
                    continue  # The broker may include an earlier boundary bar.
                start, end = session.open, session.close
            else:
                start = value.date
                if not isinstance(start, datetime) or start.tzinfo is None or start.utcoffset() is None:
                    raise HistoryError("IBKR intraday bars must have timezone-aware timestamps.")
                start = start.astimezone(timezone.utc)
                end = start + timedelta(seconds=source_seconds)
                if (start, end) not in expected:
                    if sessions[0].open <= start < sessions[-1].close:
                        raise HistoryError(f"Unexpected intraday interval for {symbol} at {start}.")
                    continue
            volume = Decimal(str(value.volume))
            if not volume.is_finite() or volume < 0 or volume != volume.to_integral_value():
                raise HistoryError(f"IBKR returned invalid share volume for {symbol} at {start}.")
            bar = Bar(symbol, start, end, value.open, value.high, value.low, value.close, int(volume))
            key = (start, end)
            if key in seen and seen[key] != bar:
                raise HistoryError(f"Conflicting duplicate IBKR bars for {symbol} at {start}.")
            seen[key] = bar
        return list(seen.values()), source_rows
