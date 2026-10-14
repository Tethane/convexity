"""Native daily Alpaca acquisition, distinct from minute-derived RTH bars."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

from convexity.runtime.types import Bar
from .alpaca import AlpacaHistoricalData, AlpacaCredentials
from .data import HistoricalData, SessionCalendar
from .history import (
    HistoricalDataset,
    HistoryError,
    HistoryRequest,
    PRICE_BASIS,
    _digest,
    validate_coverage,
)


class AlpacaDailyHistoricalData(AlpacaHistoricalData):
    """Fetch 1Day bars; never fetch years of minutes for daily ingestion.

    Native daily aggregation differs from the RTH minute adapter. This has a
    distinct cache namespace and conservatively delays availability to the next
    New York midnight; consumers must honor the availability metadata.
    """

    source = "Alpaca:SIP:split:native-daily:v1"

    async def fetch(
        self, request: HistoryRequest, *, refresh: bool = False
    ) -> HistoricalDataset:
        if not isinstance(request, HistoryRequest):
            raise TypeError("fetch requires HistoryRequest.")
        if request.bar_size != "1 day":
            raise HistoryError(
                "The native daily provider only supports 1 day; use the lazy intraday provider."
            )
        calendar = SessionCalendar.from_exchange(
            request.calendar, request.start, request.end
        )
        if not calendar.sessions:
            raise HistoryError("The requested period contains no exchange sessions.")
        eastern = ZoneInfo("America/New_York")
        available = datetime.combine(
            calendar.sessions[-1].label + timedelta(days=1),
            datetime.min.time(),
            eastern,
        )
        if available > datetime.now(timezone.utc):
            raise HistoryError(
                "Native daily bars are not available until the following New York calendar day."
            )
        credentials = self._credentials or AlpacaCredentials.load()
        params = dict(
            symbols=",".join(request.symbols),
            timeframe="1Day",
            feed="sip",
            adjustment="split",
            asof="-",
            currency="USD",
            sort="asc",
            limit=10000,
            start=str(request.start),
            end=str(request.end) + "T23:59:59-04:00",
        )
        # Date boundaries follow exchange local time, including standard time.
        params["end"] = datetime.combine(
            cast(date, request.end), datetime.max.time(), eastern
        ).isoformat()
        sessions = {s.label: s for s in calendar.sessions}
        seen = {}
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
                        if symbol not in request.symbols:
                            raise HistoryError("Alpaca returned an unexpected symbol.")
                        for row in rows:
                            stamp = datetime.fromisoformat(
                                row["t"].replace("Z", "+00:00")
                            )
                            if stamp.tzinfo is None:
                                raise HistoryError(
                                    "Alpaca daily timestamps must be timezone-aware."
                                )
                            session = sessions.get(stamp.astimezone(eastern).date())
                            if session is None:
                                continue
                            volume = Decimal(str(row["v"]))
                            if (
                                not volume.is_finite()
                                or volume < 0
                                or volume != volume.to_integral_value()
                            ):
                                raise HistoryError(
                                    "Alpaca returned invalid share volume."
                                )
                            bar = Bar(
                                symbol,
                                session.open,
                                session.close,
                                row["o"],
                                row["h"],
                                row["l"],
                                row["c"],
                                int(volume),
                            )
                            key = (symbol, session.label)
                            if key in seen and seen[key] != bar:
                                raise HistoryError(
                                    "Alpaca returned conflicting daily bars."
                                )
                            seen[key] = bar
                    pages.append(
                        dict(
                            rows=sum(len(v) for v in rows_by_symbol.values()),
                            sha256=_digest(json.loads(json.dumps(page, default=str))),
                        )
                    )
                    token = page.get("next_page_token")
                    if token is None:
                        break
                    if not isinstance(token, str) or not token or token in tokens:
                        raise HistoryError(
                            "Alpaca returned invalid or repeated pagination token."
                        )
                    tokens.add(token)
                    params["page_token"] = token
        except TimeoutError:
            raise HistoryError(
                "Alpaca daily acquisition exceeded its total timeout."
            ) from None
        except KeyError, TypeError, ValueError, AttributeError, ArithmeticError:
            raise HistoryError(
                "Alpaca returned malformed daily bars; no data was cached."
            ) from None
        if not seen:
            raise HistoryError(
                "Alpaca returned empty daily history; this does not prove permanent unavailability."
            )
        try:
            data = HistoricalData(tuple(seen.values()), self.source, PRICE_BASIS)
        except ValueError as exc:
            raise HistoryError("Alpaca returned an incomplete daily panel.") from exc
        validate_coverage(data, calendar, request)
        manifest = dict(
            provider="Alpaca",
            source=self.source,
            price_basis=PRICE_BASIS,
            request=request.to_dict(),
            feed="sip",
            adjustment="split",
            asof="-",
            source_bar_size="1 day",
            aggregation="native_daily",
            use_rth=False,
            availability="next_new_york_midnight",
            acquired_start=started,
            acquired_end=datetime.now(timezone.utc).isoformat(),
            pages=pages,
            data_identity=data.identity,
            limitations=[
                "Native daily aggregation is not minute-derived regular-session OHLCV.",
                "Conservative availability is the following New York midnight, not the labeled session close.",
                "Split-adjusted price returns; dividend cash is not included.",
                "Adjustments reflect acquisition time, not historical decision-time executable prices.",
                "After corporate-action revisions refresh all required history; mixed acquisition vintages are not certified consistent.",
            ],
        )
        return HistoricalDataset(data, calendar, manifest, Path("."))
