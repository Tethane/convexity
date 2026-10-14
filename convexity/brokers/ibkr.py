"""Read-only IBKR paper account and stock quote APIs, independent of the runtime."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import math
import os
import re
import tomllib
from typing import Any, Literal, Protocol


__all__ = [
    "PaperAccountConfig",
    "PaperAccountError",
    "AccountValue",
    "Contract",
    "Position",
    "OpenOrder",
    "PaperAccountSnapshot",
    "read_paper_account",
    "StockQuoteRequest",
    "StockQuote",
    "Quote",
    "read_stock_quote",
]


class PaperAccountError(RuntimeError):
    """The paper account could not be verified or fully read."""


@dataclass(frozen=True)
class PaperAccountConfig:
    """Explicit connection settings; timeout bounds the entire snapshot request.

    paper_session_confirmed records the caller's confirmation of the TWS/Gateway
    paper login. Account naming and a socket port alone do not prove session type.
    This first integration accepts individual paper accounts with a DU prefix, optional letters, and digits.
    """

    account: str
    port: int
    paper_session_confirmed: bool
    host: str = "127.0.0.1"
    client_id: int = 71
    timeout: float = 15.0

    def __post_init__(self) -> None:
        if not isinstance(self.account, str) or not re.fullmatch(
            r"DU[A-Z]*[0-9]+", self.account
        ):
            raise ValueError(
                "account must be the expected individual paper account ID (DU prefix, optional letters, then digits)."
            )
        if self.paper_session_confirmed is not True:
            raise ValueError(
                "Confirm the TWS/Gateway paper login, then set paper_session_confirmed = true."
            )
        if not isinstance(self.host, str) or not self.host.strip():
            raise ValueError("host must be a nonempty hostname or IP address.")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("port must be an integer between 1 and 65535.")
        if type(self.client_id) is not int or not 1 <= self.client_id <= 2**31 - 1:
            raise ValueError(
                "client_id must be a positive 32-bit integer; 0 binds manual orders and is not allowed."
            )
        if (
            type(self.timeout) not in (int, float)
            or not math.isfinite(self.timeout)
            or self.timeout <= 0
        ):
            raise ValueError("timeout must be a finite positive number of seconds.")

    @classmethod
    def from_toml(cls, path: str | Path) -> PaperAccountConfig:
        """Read [ibkr], rejecting unknown broker fields; other tables are independent."""
        with Path(path).open("rb") as stream:
            data = tomllib.load(stream)
        if not isinstance(data.get("ibkr"), dict):
            raise ValueError("Configuration must contain an [ibkr] table.")
        try:
            return cls(**data["ibkr"])
        except TypeError as exc:
            raise ValueError(f"Invalid [ibkr] configuration: {exc}") from exc

    @classmethod
    def from_env(cls) -> PaperAccountConfig:
        """Paper execution settings, supplied directly by the deployment environment.

        Historical research does not require these account permissions.
        """
        account = os.environ.get("IBKR_ACCOUNT", "")
        if not account:
            raise ValueError(
                "Set IBKR_ACCOUNT to the expected paper account, or construct PaperAccountConfig directly."
            )
        return cls(
            account=account,
            host=os.environ.get("IBKR_HOST", "127.0.0.1"),
            port=int(os.environ.get("IBKR_PORT", "7497")),
            client_id=int(os.environ.get("IBKR_CLIENT_ID", "71")),
            timeout=float(os.environ.get("IBKR_TIMEOUT", "15")),
            paper_session_confirmed=os.environ.get("IBKR_PAPER_CONFIRMED", "").lower()
            == "true",
        )


@dataclass(frozen=True)
class AccountValue:
    tag: str
    value: str
    currency: str


@dataclass(frozen=True)
class Contract:
    """Broker identity plus display fields; contract_id is the canonical identity."""

    contract_id: int
    symbol: str
    security_type: str
    local_symbol: str
    currency: str
    exchange: str
    expiry: str
    strike: Decimal
    right: str
    multiplier: str
    # (contract ID, ratio, action, exchange) for each combo leg.
    combo_legs: tuple[tuple[int, int, str, str], ...]
    primary_exchange: str = ""
    trading_class: str = ""


@dataclass(frozen=True)
class Position:
    contract: Contract
    quantity: Decimal
    average_cost: Decimal  # Broker-reported cost; not normalized per underlying unit.


@dataclass(frozen=True)
class OpenOrder:
    contract: Contract
    order_id: int
    permanent_id: int
    client_id: int
    side: str
    quantity: Decimal
    order_type: str
    time_in_force: str
    limit_price: Decimal | None
    auxiliary_price: Decimal | None
    status: str


@dataclass(frozen=True)
class PaperAccountSnapshot:
    """Detached data collected over an interval, not an atomic broker snapshot.

    Open orders are those visible to this API session across clients at request
    time. This one-shot read is not an ongoing order subscription.
    """

    account: str
    started_at: datetime
    completed_at: datetime
    summary: tuple[AccountValue, ...]
    positions: tuple[Position, ...]
    open_orders: tuple[OpenOrder, ...]


def _contract(value: Any) -> Contract:
    return Contract(
        value.conId,
        value.symbol,
        value.secType,
        value.localSymbol,
        value.currency,
        value.exchange,
        value.lastTradeDateOrContractMonth,
        Decimal(str(value.strike)),
        value.right,
        value.multiplier,
        tuple(
            (leg.conId, leg.ratio, leg.action, leg.exchange) for leg in value.comboLegs
        ),
        getattr(value, "primaryExchange", ""),
        getattr(value, "tradingClass", ""),
    )


def _price(value: float) -> Decimal | None:
    # IB's unset-double sentinel is the largest finite IEEE double.
    if not math.isfinite(value) or value == float.fromhex("0x1.fffffffffffffp+1023"):
        return None
    return Decimal(str(value))


def _new_client() -> Any:
    # Load inside the running loop: importing the optional library must not alter
    # event-loop policy or force broker dependencies on math-only users.
    try:
        from ib_async import IB
    except ImportError as exc:
        raise PaperAccountError(
            "Install the broker extra: python -m pip install -e '.[ibkr]'."
        ) from exc
    client = IB()
    # IB's hyphenated execution timestamp is explicitly UTC. ib_async 2.1.0
    # parses that format as naive and consequently applies the host timezone.
    # Normalize only this documented wire format before its existing decoder;
    # timezone-qualified replay values are left intact.
    decode_execution = client.client.decoder.handlers[11]

    def execution(fields):
        if len(fields) > 15 and re.fullmatch(
            r"[0-9]{8}-[0-9]{2}:[0-9]{2}:[0-9]{2}", fields[15]
        ):
            fields = list(fields)
            fields[15] = fields[15].replace("-", " ") + " UTC"
        return decode_execution(fields)

    client.client.decoder.handlers[11] = execution
    return client


async def _connect_paper(client: Any, config: PaperAccountConfig) -> None:
    from ib_async import StartupFetchNONE

    client.RaiseRequestErrors = True
    await client.connectAsync(
        config.host,
        config.port,
        clientId=config.client_id,
        timeout=config.timeout,
        readonly=True,
        account=config.account,
        raiseSyncErrors=True,
        fetchFields=StartupFetchNONE,
    )
    if config.account not in client.managedAccounts():
        raise PaperAccountError(
            "Expected paper account is not available in this session. Check the login and configured account."
        )


async def read_paper_account(config: PaperAccountConfig) -> PaperAccountSnapshot:
    """Connect, verify the configured account, read it, and always disconnect.

    Use ``await read_paper_account(config)`` in notebooks/async applications or
    ``asyncio.run(read_paper_account(config))`` in a synchronous script. No order
    placement, cancellation, or manual-order binding is performed.
    """
    from_context = datetime.now(timezone.utc)
    client = _new_client()
    stage = "connecting"
    try:
        async with asyncio.timeout(config.timeout):
            await _connect_paper(client, config)
            stage = "reading account summary"
            summary = await client.accountSummaryAsync(config.account)
            values = tuple(
                AccountValue(v.tag, v.value, v.currency)
                for v in summary
                if v.account == config.account
            )
            if not values:
                raise PaperAccountError(
                    "Broker returned no summary for the expected account; snapshot is incomplete."
                )
            stage = "reading positions"
            positions = await client.reqPositionsAsync()
            owned = tuple(
                Position(
                    _contract(p.contract),
                    Decimal(str(p.position)),
                    Decimal(str(p.avgCost)),
                )
                for p in positions
                if p.account == config.account
            )
            stage = "reading open orders"
            trades = await client.reqAllOpenOrdersAsync()
            orders = tuple(
                OpenOrder(
                    _contract(t.contract),
                    t.order.orderId,
                    t.order.permId,
                    t.order.clientId,
                    t.order.action,
                    Decimal(str(t.order.totalQuantity)),
                    t.order.orderType,
                    t.order.tif,
                    _price(t.order.lmtPrice),
                    _price(t.order.auxPrice),
                    t.orderStatus.status,
                )
                for t in trades
                if t.order.account == config.account
            )
            if not client.isConnected():
                raise PaperAccountError(
                    "Broker disconnected during the snapshot; retry after reconnecting."
                )
            return PaperAccountSnapshot(
                config.account,
                from_context,
                datetime.now(timezone.utc),
                values,
                owned,
                orders,
            )
    except TimeoutError as exc:
        raise PaperAccountError(
            f"Timed out while {stage}. Check paper login, API access, port and client ID; retry or increase timeout."
        ) from exc
    except PaperAccountError:
        raise
    except Exception as exc:
        raise PaperAccountError(
            f"Failed while {stage} ({type(exc).__name__}). Check TWS/Gateway paper login, socket API settings, host/port and client ID."
        ) from exc
    finally:
        client.disconnect()


_MARKET_DATA_TYPES = {"live": 1, "frozen": 2, "delayed": 3, "delayed_frozen": 4}


@dataclass(frozen=True)
class StockQuoteRequest:
    """Resolve one stock/ETF, then request a non-regulatory quote snapshot.

    max_age bounds local bid/ask receipt age, not exchange-source age. Delayed and
    frozen data are always labeled non_live regardless of receipt age.
    """

    symbol: str
    exchange: str = "SMART"
    currency: str = "USD"
    primary_exchange: str = ""
    market_data_type: Literal["live", "frozen", "delayed", "delayed_frozen"] = "live"
    max_age: float = 30.0

    def __post_init__(self) -> None:
        for name in ("symbol", "exchange", "currency"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value.strip()
                or value != value.strip()
            ):
                raise ValueError(
                    f"{name} must be a nonempty string without surrounding whitespace."
                )
        if not isinstance(self.primary_exchange, str):
            raise ValueError("primary_exchange must be a string.")
        if self.market_data_type not in _MARKET_DATA_TYPES:
            raise ValueError(
                "market_data_type must be live, frozen, delayed, or delayed_frozen."
            )
        if (
            type(self.max_age) not in (int, float)
            or not math.isfinite(self.max_age)
            or self.max_age <= 0
        ):
            raise ValueError("max_age must be a finite positive number of seconds.")


@dataclass(frozen=True)
class Quote:
    """Detached quote; receipt timestamps do not prove exchange-source freshness.

    status=received means both sides were recently received on a live feed, not
    that the quote is executable. Check again before any later order submission.
    """

    contract: Contract
    requested_data_type: str
    market_data_type: str
    bid: Decimal | None
    ask: Decimal | None
    bid_size: Decimal | None
    ask_size: Decimal | None
    bid_received_at: datetime | None
    ask_received_at: datetime | None
    completed_at: datetime
    max_age: float
    snapshot_complete: bool
    status: Literal[
        "received", "missing", "stale", "non_live", "invalid", "unknown", "incomplete"
    ]
    diagnostics: tuple[str, ...]


# Compatibility name retained for existing stock clients.
StockQuote = Quote


def _positive(value: float) -> Decimal | None:
    result = _price(value)
    return result if result is not None and result > 0 else None


class _QuoteSettings(Protocol):
    @property
    def market_data_type(self) -> str: ...
    @property
    def max_age(self) -> float: ...


def _stock_quote(
    contract: Any,
    request: _QuoteSettings,
    ticker: Any,
    receipts: dict[str, datetime],
    complete: bool,
    diagnostics: list[str],
) -> StockQuote:
    now = datetime.now(timezone.utc)
    bid = _positive(ticker.bid) if ticker else None
    ask = _positive(ticker.ask) if ticker else None
    bid_size = _positive(ticker.bidSize) if ticker else None
    ask_size = _positive(ticker.askSize) if ticker else None
    data_type = next(
        (
            k
            for k, v in _MARKET_DATA_TYPES.items()
            if ticker is not None and ticker.marketDataType == v
        ),
        "unknown",
    )
    status = "received"
    if not complete:
        status = "incomplete"
    elif bid is None or ask is None or bid_size is None or ask_size is None:
        status = "missing"
        diagnostics.append(
            "No complete two-sided quote with positive sizes was returned."
        )
    elif bid > ask:
        status = "invalid"
        diagnostics.append("Crossed bid/ask; do not use this quote for execution.")
    elif data_type in ("frozen", "delayed", "delayed_frozen"):
        status = "non_live"
        diagnostics.append("Delayed/frozen prices are not current executable quotes.")
    elif data_type == "unknown" or not all(k in receipts for k in ("bid", "ask")):
        status = "unknown"
        diagnostics.append("Feed type or bid/ask receipt timestamps are unavailable.")
    elif any(
        not 0 <= (now - receipts[k]).total_seconds() <= request.max_age
        for k in ("bid", "ask")
    ):
        status = "stale"
        diagnostics.append(
            "Bid or ask receipt age exceeds max_age or the local clock moved backward."
        )
    return StockQuote(
        _contract(contract),
        request.market_data_type,
        data_type,
        bid,
        ask,
        bid_size,
        ask_size,
        receipts.get("bid"),
        receipts.get("ask"),
        now,
        request.max_age,
        complete,
        status,
        tuple(diagnostics),
    )


async def _read_quote(
    config: PaperAccountConfig, request: Any, make_contract: Any, security_type: str
) -> Quote:
    """Verify the paper account, resolve a unique stock, read bid/ask, disconnect.

    Missing/denied/timed-out market data returns a diagnostic StockQuote. Failure
    to connect, verify the account, or uniquely resolve a stock raises
    PaperAccountError. The total operation is bounded by config.timeout.
    """
    client = _new_client()
    stage = "connecting"
    resolved = None
    ticker = None
    receipts: dict[str, datetime] = {}
    diagnostics: list[str] = []
    complete = False

    def on_ticks(tickers: Any) -> None:
        nonlocal ticker
        for value in tickers:
            if resolved is not None and value.contract.conId == resolved.conId:
                ticker = value
                for tick in value.ticks:
                    if tick.tickType in (1, 66):
                        receipts["bid"] = tick.time
                    elif tick.tickType in (2, 67):
                        receipts["ask"] = tick.time

    def on_error(req_id: int, code: int, message: str, contract: Any) -> None:
        if (
            resolved is not None
            and contract is not None
            and contract.conId == resolved.conId
        ):
            diagnostics.append(
                f"IBKR {code}: {message.replace(config.account, '[ACCOUNT]')}"
            )

    try:
        from ib_async import RequestError

        client.pendingTickersEvent += on_ticks
        client.errorEvent += on_error
        try:
            async with asyncio.timeout(config.timeout):
                await _connect_paper(client, config)
                stage = "resolving contract"
                stock = make_contract()
                details = await client.reqContractDetailsAsync(stock)
                if len(details) != 1:
                    raise PaperAccountError(
                        f"Contract resolution returned {len(details)} matches. Check symbol, currency, exchange and primary_exchange."
                    )
                resolved = details[0].contract
                if resolved.secType != security_type or resolved.conId <= 0:
                    raise PaperAccountError(
                        "Broker did not resolve a valid contract of the requested type."
                    )
                resolved.exchange = request.exchange
                stage = "reading quote"
                client.reqMarketDataType(_MARKET_DATA_TYPES[request.market_data_type])
                values = await client.reqTickersAsync(
                    resolved, regulatorySnapshot=False
                )
                if len(values) != 1:
                    raise PaperAccountError(
                        "Broker returned an unexpected number of quote snapshots."
                    )
                ticker = values[0]
                complete = True
                if not client.isConnected():
                    raise PaperAccountError(
                        "Broker disconnected during the quote snapshot."
                    )
        except (TimeoutError, RequestError) as exc:
            if stage != "reading quote":
                raise PaperAccountError(
                    f"Failed while {stage} ({type(exc).__name__}). Check connection settings and stock identity."
                ) from exc
            if isinstance(exc, TimeoutError):
                diagnostics.append(
                    "Quote snapshot timed out; partial data is not a completed quote. Check market-data permissions/session or increase timeout."
                )
            elif not diagnostics:
                diagnostics.append(
                    f"IBKR {exc.code}: {exc.message.replace(config.account, '[ACCOUNT]')}"
                )
        assert resolved is not None
        return _stock_quote(resolved, request, ticker, receipts, complete, diagnostics)
    except PaperAccountError:
        raise
    except Exception as exc:
        raise PaperAccountError(
            f"Failed while {stage} ({type(exc).__name__}). Check Gateway/API access and market-data permissions."
        ) from exc
    finally:
        client.pendingTickersEvent -= on_ticks
        client.errorEvent -= on_error
        client.disconnect()


async def read_stock_quote(
    config: PaperAccountConfig, request: StockQuoteRequest
) -> Quote:
    """Verify account, resolve one stock and return a labeled quote snapshot."""
    from ib_async import Stock

    return await _read_quote(
        config,
        request,
        lambda: Stock(
            request.symbol,
            request.exchange,
            request.currency,
            primaryExchange=request.primary_exchange,
        ),
        "STK",
    )
