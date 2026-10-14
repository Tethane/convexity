"""Read-only TWS/Gateway history, independent of paper trading and backtests."""

import asyncio
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import ClassVar, cast

import pandas as pd

from convexity._ibkr_connection import connect_readonly
from ._http import positive
from .core import OHLCV, DataError, empty_frame

_LOG = logging.getLogger(__name__)


def _window(bar_seconds: int) -> int:
    """Bound minute-or-larger requests to 2,000 potential bars and 365 D.

    Count a full 24-hour market, not stock regular-session hours: the same
    adapter accepts FX, futures, extended hours and other explicit contracts.
    Use whole days so IB's rounded duration cannot exceed the row budget.
    """
    return min(365, max(1, 2000 * bar_seconds // 86400)) * 86400


def _chunks(start: pd.Timestamp, end: pd.Timestamp, window: int):
    """Newest first, covering [start, end) with no gaps, including leap days.

    IB durations use whole seconds/days, so the final request can overfetch.
    Shared DataFrame normalization clips and deduplicates those observations.
    """
    cursor = end
    while cursor > start:
        lower = max(start, cursor - pd.Timedelta(seconds=window))
        seconds = math.ceil((cursor - lower).total_seconds())
        duration = (
            f"{math.ceil(seconds / 86400)} D" if seconds >= 86400 else f"{seconds} S"
        )
        yield lower, cursor, duration
        cursor = lower


class IBKR:
    """Stock symbols by default; supply contracts={label: ib_async.Contract(...)}
    for other instruments or to disambiguate an exchange/currency.

    Each symbol query owns a fresh read-only connection and always disconnects.
    Use different client IDs for concurrent queries against the same Gateway.
    """

    name = "ibkr"
    # Current IB duration limits permit up to 365 D for minute-or-larger bars.
    # Keep each response modest rather than requesting the maximum duration
    # indiscriminately. Subminute bars retain their stricter windows and pacing.
    # https://www.interactivebrokers.com/docs/tws-api/doc/market-data-historical/historical-bars/max-duration-per-bar-size
    intervals: ClassVar[dict[str, tuple[str, int]]] = {
        "1s": ("1 secs", 1800),
        "5s": ("5 secs", 3600),
        "10s": ("10 secs", 14400),
        "15s": ("15 secs", 14400),
        "30s": ("30 secs", 28800),
        "1m": ("1 min", _window(60)),
        "2m": ("2 mins", _window(120)),
        "3m": ("3 mins", _window(180)),
        "5m": ("5 mins", _window(300)),
        "10m": ("10 mins", _window(600)),
        "15m": ("15 mins", _window(900)),
        "20m": ("20 mins", _window(1200)),
        "30m": ("30 mins", _window(1800)),
        "1h": ("1 hour", _window(3600)),
        "2h": ("2 hours", _window(7200)),
        "3h": ("3 hours", _window(10800)),
        "4h": ("4 hours", _window(14400)),
        "8h": ("8 hours", _window(28800)),
        "1d": ("1 day", _window(86400)),
    }

    def __init__(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        client_id: int = 71,
        timeout: float = 60,
        request_interval: float = 0.35,
        exchange: str = "SMART",
        currency: str = "USD",
        use_rth: bool = True,
        what_to_show: str = "TRADES",
        contracts: dict | None = None,
    ):
        positive(timeout, "timeout")
        positive(request_interval, "request_interval", zero=True)
        host = host if host is not None else os.environ.get("IBKR_HOST")
        port = (
            port
            if port is not None
            else (int(os.environ["IBKR_PORT"]) if "IBKR_PORT" in os.environ else None)
        )
        if host is not None and (not isinstance(host, str) or not host.strip()):
            raise ValueError("host must be a nonempty hostname or IP address.")
        if port is not None and (type(port) is not int or not 1 <= port <= 65535):
            raise ValueError("port must be between 1 and 65535.")
        if type(client_id) is not int or client_id <= 0:
            raise ValueError("client_id must be a positive integer.")
        self.host, self.port, self.client_id = host, port, client_id
        self.timeout, self.request_interval = timeout, request_interval
        self.exchange, self.currency = exchange, currency
        self.use_rth, self.what_to_show = use_rth, what_to_show
        self.contracts = dict(contracts or {})

    def read(self, symbol, start, end, interval):
        if interval not in self.intervals:
            raise ValueError(f"IBKR intervals: {', '.join(self.intervals)}")
        if interval == "1d" and (start != start.normalize() or end != end.normalize()):
            raise ValueError("IBKR daily boundaries must be midnight UTC dates.")
        # A notebook already has an event loop. Own the IB connection and loop
        # together in a worker instead of patching global asyncio behavior.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._read(symbol, start, end, interval))
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(
                lambda: asyncio.run(self._read(symbol, start, end, interval))
            ).result()

    async def _read(self, symbol, start, end, interval):
        try:
            from ib_async import IB, Stock
        except ImportError as exc:
            raise ImportError(
                "Install IBKR support: pip install 'convexity[ibkr,data]'"
            ) from exc
        client = IB()
        client.RaiseRequestErrors = True
        rows = []
        begun = time.monotonic()
        try:
            async with asyncio.timeout(self.timeout):
                await connect_readonly(
                    client,
                    host=self.host,
                    port=self.port,
                    client_id=self.client_id,
                    timeout=self.timeout,
                )
                contract = self.contracts.get(symbol)
                if contract is None:
                    contract = Stock(symbol, self.exchange, self.currency)
                details = await client.reqContractDetailsAsync(contract)
                resolved = details[0].contract if len(details) == 1 else None
                if resolved is None or resolved.conId <= 0:
                    raise DataError(
                        f"IBKR could not uniquely resolve {symbol}; supply an explicit contract."
                    )
                # Retain the requested routing exchange (e.g. SMART).
                resolved.exchange = contract.exchange
            bar_size, window = self.intervals[interval]
            count = math.ceil((end - start).total_seconds() / window)
            _LOG.info(
                "IBKR %s: fetching %s bars in %d requests", symbol, interval, count
            )
            for completed, (_, cursor, duration) in enumerate(
                _chunks(start, end, window), 1
            ):
                if completed > 1:
                    # BID_ASK consumes two requests under the small-bar quota.
                    small_bar_delay = 20.1 if self.what_to_show == "BID_ASK" else 10.1
                    await asyncio.sleep(
                        max(self.request_interval, small_bar_delay)
                        if interval.endswith("s")
                        else self.request_interval
                    )
                # Disable ib_async's timeout-to-empty conversion; a timeout is
                # a failed request, never evidence of a legitimate data gap.
                async with asyncio.timeout(self.timeout):
                    bars = await client.reqHistoricalDataAsync(
                        resolved,
                        endDateTime=cast(datetime, cursor.to_pydatetime()),
                        durationStr=duration,
                        barSizeSetting=bar_size,
                        whatToShow=self.what_to_show,
                        useRTH=self.use_rth,
                        formatDate=2,
                        keepUpToDate=False,
                        timeout=0,
                    )
                if not client.isConnected():
                    raise DataError("IBKR disconnected during history acquisition.")
                for bar in bars:
                    stamp = pd.Timestamp(bar.date)
                    if interval != "1d" and stamp.tzinfo is None:
                        raise DataError(
                            "IBKR returned an intraday timestamp without a timezone."
                        )
                    stamp = (
                        stamp.tz_localize("UTC")
                        if stamp.tzinfo is None
                        else stamp.tz_convert("UTC")
                    )
                    rows.append(
                        [
                            stamp,
                            bar.open,
                            bar.high,
                            bar.low,
                            bar.close,
                            float("nan") if bar.volume < 0 else bar.volume,
                        ]
                    )
                _LOG.info(
                    "IBKR %s: request %d/%d complete, %d bars received",
                    symbol,
                    completed,
                    count,
                    len(bars),
                )
            if rows:
                frame = pd.DataFrame(rows, columns=["timestamp", *OHLCV]).set_index(
                    "timestamp"
                )
            else:
                frame = empty_frame()
            frame.attrs = {
                "contract_id": resolved.conId,
                "security_type": resolved.secType,
                "currency": resolved.currency,
                "exchange": resolved.exchange,
                "what_to_show": self.what_to_show,
                "use_rth": self.use_rth,
                "price_basis": "ibkr_" + self.what_to_show.lower(),
                "volume_unit": "provider_native",
                "request_count": count,
                "elapsed_seconds": time.monotonic() - begun,
                "timestamp_kind": "session_date" if interval == "1d" else "bar_open",
            }
            return frame
        except DataError:
            raise
        except Exception as exc:
            raise DataError(
                f"IBKR history failed for {symbol} ({type(exc).__name__}): {exc}"
            ) from exc
        finally:
            client.disconnect()
