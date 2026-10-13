"""Alpaca SIP history, normalized to complete regular exchange sessions."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
import math
import os
from pathlib import Path
import stat
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

from convexity.runtime.types import Bar
from .bar_size import parse_bar_size
from .data import HistoricalData, SessionCalendar
from .history import (HistoricalDataset, HistoryError, HistoryRequest, PRICE_BASIS,
                      _aggregate, _digest, validate_coverage)


def credentials_path() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "convexity" / "alpaca.json"


@dataclass(frozen=True)
class AlpacaCredentials:
    """Secrets are excluded from repr; environment variables override the local file."""

    api_key: str = field(repr=False)
    secret_key: str = field(repr=False)

    def __post_init__(self):
        if any(not isinstance(value, str) or not value.strip() or any(c.isspace() for c in value)
               for value in (self.api_key, self.secret_key)):
            raise HistoryError("Alpaca requires a nonempty API key and secret without whitespace.")

    @classmethod
    def load(cls) -> AlpacaCredentials:
        key, secret = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
        if key is not None or secret is not None:
            if not key or not secret:
                raise HistoryError("Set both APCA_API_KEY_ID and APCA_API_SECRET_KEY.")
            return cls(key, secret)
        path = credentials_path()
        try:
            with path.open() as stream:
                info = os.fstat(stream.fileno())
                if os.name == "posix" and (info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077):
                    raise HistoryError("Alpaca credential file must be owned by you with mode 600.")
                values = json.load(stream)
            return cls(values["api_key"], values["secret_key"])
        except (OSError, ValueError, KeyError, TypeError):
            raise HistoryError("Alpaca credentials unavailable. Run python scripts/configure_alpaca.py "
                               "or set APCA_API_KEY_ID and APCA_API_SECRET_KEY.") from None


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward credential headers to a redirected host.
        return None


class AlpacaHistoricalData:
    """Read-only REST acquisition; Research supplies the persistent SQLite cache.

    Explicit SIP, split adjustment, fixed symbol mapping and regular-session
    minute aggregation keep the feed's semantics stable across requests. Even
    daily requests use minutes: native daily bars have different trade/session
    rules. Sub-minute requests are unsupported and can use the IBKR fallback.
    """

    source = "Alpaca:SIP:split:RTH:historical:v1"
    price_basis = PRICE_BASIS

    def __init__(self, credentials: AlpacaCredentials | None = None, *,
                 request_timeout: float = 15, total_timeout: float = 120,
                 request_interval: float = 0.31):
        for name, value in (("request_timeout", request_timeout), ("total_timeout", total_timeout),
                            ("request_interval", request_interval)):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (name != "request_interval" and value == 0):
                raise ValueError(f"Invalid {name}.")
        self._credentials = credentials
        self.request_timeout = request_timeout
        self.total_timeout = total_timeout
        self.request_interval = request_interval

    def _request_page(self, params: dict, credentials: AlpacaCredentials) -> dict:
        request = Request("https://data.alpaca.markets/v2/stocks/bars?" + urlencode(params),
                          headers={"APCA-API-KEY-ID": credentials.api_key,
                                   "APCA-API-SECRET-KEY": credentials.secret_key,
                                   "Accept": "application/json"})
        with build_opener(_NoRedirect()).open(request, timeout=self.request_timeout) as response:
            return json.load(response, parse_float=Decimal)

    async def _page(self, params: dict, credentials: AlpacaCredentials) -> dict:
        for attempt in range(3):
            try:
                return await asyncio.to_thread(self._request_page, params, credentials)
            except HTTPError as exc:
                code = exc.code
                retry_after = exc.headers.get("Retry-After", "") if exc.headers else ""
                exc.close()
                if (code == 429 or 500 <= code < 600) and attempt < 2:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = 2 ** attempt
                    await asyncio.sleep(min(5, max(self.request_interval, delay)))
                    continue
                # Do not include response bodies, URLs, headers, or exception reprs.
                raise HistoryError(f"Alpaca historical request failed (HTTP {code}); check credentials, "
                                   "SIP entitlement, rate limits and service availability.") from None
            except (URLError, OSError, ValueError):
                raise HistoryError("Alpaca historical request failed (network, timeout or invalid JSON).") from None
        raise AssertionError("Unreachable")

    async def fetch(self, request: HistoryRequest, *, refresh: bool = False) -> HistoricalDataset:
        if not isinstance(request, HistoryRequest):
            raise TypeError("fetch requires HistoryRequest.")
        spec = parse_bar_size(request.bar_size)
        if spec.unit == "second" and spec.count % 60:
            raise HistoryError("Alpaca stock bars do not support sub-minute resolution; use IBKR.")
        calendar = SessionCalendar.from_exchange(request.calendar, request.start, request.end)
        if not calendar.sessions:
            raise HistoryError("The requested period contains no exchange sessions.")
        if calendar.sessions[-1].close > datetime.now(timezone.utc):
            raise HistoryError("The requested period includes an unfinished or future session.")
        credentials = self._credentials or AlpacaCredentials.load()
        params = dict(symbols=",".join(request.symbols), timeframe="1Min", feed="sip", adjustment="split",
                      asof="-", currency="USD", sort="asc", limit=10000,
                      start=calendar.sessions[0].open.isoformat(),
                      end=(calendar.sessions[-1].close - timedelta(microseconds=1)).isoformat())
        sessions = {s.open.date(): s for s in calendar.sessions}
        seen: dict[str, dict[datetime, Bar]] = {symbol: {} for symbol in request.symbols}
        pages = []
        tokens = set()
        started = datetime.now(timezone.utc).isoformat()
        try:
            async with asyncio.timeout(self.total_timeout):
                while True:
                    if pages:
                        await asyncio.sleep(self.request_interval)
                    page = await self._page(params, credentials)
                    rows_by_symbol = page["bars"] or {}
                    for symbol, rows in rows_by_symbol.items():
                        if symbol not in seen:
                            raise HistoryError("Alpaca returned an unexpected symbol.")
                        for row in rows:
                            start = datetime.fromisoformat(row["t"].replace("Z", "+00:00"))
                            if start.tzinfo is None or start.utcoffset() is None:
                                raise HistoryError("Alpaca timestamps must be timezone-aware.")
                            start = start.astimezone(timezone.utc)
                            session = sessions.get(start.date())
                            if session is None or not session.open <= start < session.close:
                                continue  # Discard extended hours and non-session observations.
                            if start.second or start.microsecond:
                                raise HistoryError("Alpaca returned a misaligned minute bar.")
                            volume = Decimal(str(row["v"]))
                            if not volume.is_finite() or volume < 0 or volume != volume.to_integral_value():
                                raise HistoryError("Alpaca returned invalid share volume.")
                            bar = Bar(symbol, start, start + timedelta(minutes=1),
                                      row["o"], row["h"], row["l"], row["c"], int(volume))
                            previous = seen[symbol].get(start)
                            if previous is not None and previous != bar:
                                raise HistoryError("Alpaca returned conflicting duplicate bars.")
                            seen[symbol][start] = bar
                    # Hash normalized JSON (Decimal strings preserve exact wire prices).
                    pages.append(dict(rows=sum(len(rows) for rows in rows_by_symbol.values()),
                                      sha256=_digest(json.loads(json.dumps(page, default=str)))))
                    token = page.get("next_page_token")
                    if token is None:
                        break
                    if not isinstance(token, str) or not token or token in tokens:
                        raise HistoryError("Alpaca returned an invalid or repeated pagination token.")
                    tokens.add(token)
                    params["page_token"] = token
        except TimeoutError:
            raise HistoryError("Alpaca historical acquisition exceeded its total timeout.") from None
        except (KeyError, TypeError, ValueError, AttributeError, ArithmeticError):
            raise HistoryError("Alpaca returned malformed historical bars; no data was cached.") from None
        if any(not rows for rows in seen.values()):
            raise HistoryError("Alpaca returned empty history for a requested symbol.")
        try:
            minute_data = HistoricalData(tuple(bar for rows in seen.values() for bar in rows.values()), self.source, PRICE_BASIS)
        except ValueError:
            raise HistoryError("Alpaca returned incomplete or overlapping bar panels.") from None
        validate_coverage(minute_data, calendar, HistoryRequest(request.symbols, request.start, request.end, "1 min", request.calendar))
        data = minute_data if request.bar_size == "1 min" else HistoricalData(
            tuple(bar for rows in seen.values() for bar in _aggregate(list(rows.values()), calendar.sessions, request.bar_size)),
            self.source, PRICE_BASIS)
        validate_coverage(data, calendar, request)
        manifest = dict(provider="Alpaca", source=self.source, price_basis=PRICE_BASIS,
                        request=request.to_dict(), feed="sip", adjustment="split", asof="-", use_rth=True,
                        source_bar_size="1 min", aggregation="session_calendar_ohlcv",
                        acquired_start=started, acquired_end=datetime.now(timezone.utc).isoformat(),
                        pages=pages, data_identity=data.identity,
                        limitations=["Split-adjusted price returns; no dividend cash flows.",
                                     "Fixed symbols; no historical ticker mapping or survivorship correction.",
                                     "Missing minutes fail coverage validation; prices are never fabricated."])
        return HistoricalDataset(data, calendar, manifest, Path("."))
