"""Observed stock history and broker-supplied liquid-session schedules."""

import asyncio
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

from convexity.backtest import HistoricalData, Session, SessionCalendar
from convexity.runtime import Bar
from .ibkr import PaperAccountError
from .ibkr_trading import PaperBroker


async def _stock_detail(broker: PaperBroker, symbol: str):
    from ib_async import Stock

    broker._check()
    details = await broker._ib.reqContractDetailsAsync(Stock(symbol, "SMART", "USD"))
    if len(details) != 1 or details[0].contract.secType != "STK":
        raise PaperAccountError("Stock data request must resolve uniquely.")
    return details[0]


async def read_stock_history(
    broker: PaperBroker, symbol: str, *, seconds: int = 1800
) -> HistoricalData:
    """Read completed one-minute trade bars; drop the still-forming last bar."""
    if type(seconds) is not int or not 180 <= seconds <= 366 * 86400:
        raise ValueError("History window must be between 180 seconds and 366 days.")
    async with asyncio.timeout(broker.config.timeout):
        detail = await _stock_detail(broker, symbol)
    requested_at = datetime.now(timezone.utc)
    values = []
    remaining = seconds
    offset = 0
    while remaining:
        duration = min(86400, remaining)
        # Keep one-minute requests bounded. Empty holiday/weekend slices are
        # allowed; coverage is checked against actual exchange sessions above.
        async with asyncio.timeout(broker.config.timeout):
            chunk = await broker._ib.reqHistoricalDataAsync(
                detail.contract,
                endDateTime=""
                if offset == 0
                else requested_at - timedelta(seconds=offset),
                durationStr=f"{duration} S",
                barSizeSetting="1 min",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=2,
                keepUpToDate=False,
            )
        values.extend(chunk)
        offset += duration
        remaining -= duration
    now = datetime.now(timezone.utc)
    bars = {}
    for value in values:
        if not isinstance(value.date, datetime) or value.date.tzinfo is None:
            raise PaperAccountError(
                "Broker did not return timezone-aware intraday bars."
            )
        end = value.date + timedelta(minutes=1)
        if end > now:
            continue
        volume = Decimal(str(value.volume))
        if not volume.is_finite() or volume < 0 or volume != volume.to_integral_value():
            raise PaperAccountError("Stock bar volume is missing or not whole shares.")
        bar = Bar(
            symbol,
            value.date,
            end,
            value.open,
            value.high,
            value.low,
            value.close,
            int(volume),
        )
        if value.date in bars and bars[value.date] != bar:
            raise PaperAccountError(
                "Overlapping historical slices disagree on an observed bar."
            )
        bars[value.date] = bar
    if not bars:
        raise PaperAccountError("No completed stock bars were returned.")
    return HistoricalData(
        tuple(bars[stamp] for stamp in sorted(bars)),
        source=f"IBKR:{symbol}:TRADES:1min",
        price_basis="split_adjusted_price_return",
    )


async def read_stock_sessions(broker: PaperBroker, symbol: str) -> SessionCalendar:
    """Read the limited liquidHours horizon supplied by the broker.

    This is not a full-month exchange calendar. Persist/import a complete table
    for production monthly first/last-session schedules.
    """
    async with asyncio.timeout(broker.config.timeout):
        detail = await _stock_detail(broker, symbol)
    zone = ZoneInfo(detail.timeZoneId)
    sessions = []
    for row in detail.liquidHours.split(";"):
        if not row or "CLOSED" in row:
            continue
        if "," in row:
            raise PaperAccountError(
                "Split liquid sessions are unsupported by this stock calendar."
            )
        begin, end = row.split("-")
        label, begin_time = begin.split(":")
        if ":" in end:
            end_label, end_time = end.split(":")
        else:
            end_label, end_time = label, end
        opening = datetime.strptime(label + begin_time, "%Y%m%d%H%M").replace(
            tzinfo=zone
        )
        closing = datetime.strptime(end_label + end_time, "%Y%m%d%H%M").replace(
            tzinfo=zone
        )
        sessions.append(Session(date.fromisoformat(label), opening, closing))
    if not sessions:
        raise PaperAccountError("Broker supplied no usable liquid sessions.")
    return SessionCalendar(tuple(sessions), name=f"IBKR:{symbol}:liquidHours")


async def read_session_option_quotes(broker: PaperBroker, requests) -> tuple:
    """Resolve declared option subscriptions using the execution connection.

    A second connection with the same client ID would disrupt execution. Keep
    receipt timestamps and exact contract checks on the verified shared session.
    """
    from ib_async import Option
    from . import ibkr

    broker._check()
    receipts = {}

    def ticks(tickers):
        for ticker in tickers:
            if ticker.contract.conId not in receipts:
                continue
            for tick in ticker.ticks:
                if tick.tickType in (1, 66):
                    receipts[ticker.contract.conId]["bid"] = tick.time
                elif tick.tickType in (2, 67):
                    receipts[ticker.contract.conId]["ask"] = tick.time

    async with asyncio.timeout(broker.config.timeout):
        contracts = []
        for request in requests:
            details = await broker._ib.reqContractDetailsAsync(
                Option(
                    request.symbol,
                    request.expiry.strftime("%Y%m%d"),
                    float(request.strike),
                    request.right,
                    request.exchange,
                    multiplier=request.multiplier,
                    currency=request.currency,
                    tradingClass=request.trading_class,
                )
            )
            if len(details) != 1:
                raise PaperAccountError("Option subscription must resolve uniquely.")
            contract = details[0].contract
            actual = (
                contract.symbol,
                contract.lastTradeDateOrContractMonth[:8],
                Decimal(str(contract.strike)),
                contract.right,
                contract.multiplier,
                contract.currency,
                contract.tradingClass,
            )
            expected = (
                request.symbol,
                request.expiry.strftime("%Y%m%d"),
                request.strike,
                request.right,
                request.multiplier,
                request.currency,
                request.trading_class,
            )
            if contract.secType != "OPT" or contract.conId <= 0 or actual != expected:
                raise PaperAccountError(
                    "Resolved option differs from its subscription."
                )
            contracts.append(contract)
        if len({contract.conId for contract in contracts}) != len(contracts):
            raise ValueError("Option subscriptions must identify distinct contracts.")
        receipts = {contract.conId: {} for contract in contracts}
        broker._ib.pendingTickersEvent += ticks
        try:
            # Trading subscriptions always require live observations. Delayed
            # entitlements remain visible in each returned quote's status.
            if any(request.market_data_type != "live" for request in requests):
                raise ValueError(
                    "Paper trading option subscriptions require live market data."
                )
            broker._ib.reqMarketDataType(1)
            tickers = await broker._ib.reqTickersAsync(
                *contracts, regulatorySnapshot=False
            )
            if len(tickers) != len(contracts) or any(
                ticker.contract.conId != contract.conId
                for ticker, contract in zip(tickers, contracts)
            ):
                raise PaperAccountError(
                    "Option snapshots did not cover every subscription in the expected order."
                )
            return tuple(
                ibkr._stock_quote(
                    contract, request, ticker, receipts[contract.conId], True, []
                )
                for request, contract, ticker in zip(requests, contracts, tickers)
            )
        finally:
            broker._ib.pendingTickersEvent -= ticks
