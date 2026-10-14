import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace as NS

import pytest

from convexity.brokers import ibkr
from tests.brokers.test_ibkr import FakeClient, config, contract, pytestmark


class Event:
    def __init__(self):
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self

    def __isub__(self, handler):
        self.handlers.remove(handler)
        return self

    def emit(self, *args):
        for handler in self.handlers:
            handler(*args)


class QuoteClient(FakeClient):
    def __init__(self):
        super().__init__()
        self.pendingTickersEvent = Event()
        self.errorEvent = Event()
        self.details = [NS(contract=contract())]
        self.data_type = 1
        self.bid = 600.25
        self.ask = 600.5
        self.size = 10
        self.age = 0
        self.send_bid = True
        self.send_ask = True
        self.error = False
        self.hang = False

    async def reqContractDetailsAsync(self, stock):
        self.calls.append("resolve")
        assert stock.secType == "STK"
        return self.details

    def reqMarketDataType(self, value):
        self.requested_type = value

    async def reqTickersAsync(self, resolved, *, regulatorySnapshot):
        from ib_async import RequestError

        assert regulatorySnapshot is False
        self.calls.append("quote")
        if self.error:
            self.errorEvent.emit(8, 10089, "Missing subscription for DU123", resolved)
            raise RequestError(8, 10089, "Missing subscription for DU123")
        now = datetime.now(timezone.utc) - timedelta(seconds=self.age)
        ticks = [NS(tickType=4, time=datetime.now(timezone.utc))]
        if self.send_bid:
            ticks.append(NS(tickType=1, time=now))
        if self.send_ask:
            ticks.append(NS(tickType=2, time=now))
        ticker = NS(
            contract=resolved,
            bid=self.bid,
            ask=self.ask,
            bidSize=self.size,
            askSize=self.size,
            marketDataType=self.data_type,
            ticks=ticks,
        )
        self.pendingTickersEvent.emit([ticker])
        if self.hang:
            await asyncio.Event().wait()
        return [ticker]


@pytest.fixture
def client(monkeypatch):
    value = QuoteClient()
    monkeypatch.setattr(ibkr, "_new_client", lambda: value)
    return value


def read(**kwargs):
    return asyncio.run(
        ibkr.read_stock_quote(config(), ibkr.StockQuoteRequest("SPY", **kwargs))
    )


def test_live_quote_and_cleanup(client):
    quote = read()
    assert quote.status == "received"
    assert quote.snapshot_complete
    assert quote.bid == Decimal("600.25")
    assert quote.ask == Decimal("600.5")
    assert quote.contract.contract_id == 123
    assert quote.bid_received_at is not None
    assert quote.market_data_type == "live"
    assert client.requested_type == 1
    assert client.disconnected
    assert not client.pendingTickersEvent.handlers and not client.errorEvent.handlers


@pytest.mark.parametrize("data_type", [2, 3, 4])
def test_nonlive_is_not_recent_live_quote(client, data_type):
    client.data_type = data_type
    quote = read(market_data_type="delayed_frozen")
    assert client.requested_type == 4
    assert quote.status == "non_live"
    assert quote.market_data_type in ("frozen", "delayed", "delayed_frozen")


def test_actual_feed_type_overrides_requested_type(client):
    assert read(market_data_type="delayed").market_data_type == "live"


@pytest.mark.parametrize(
    "value",
    [-1, 0, float("nan"), float("inf"), float.fromhex("0x1.fffffffffffffp+1023")],
)
def test_missing_stock_price_sentinels(client, value):
    client.bid = value
    quote = read()
    assert quote.bid is None
    assert quote.status == "missing"


def test_zero_size_is_missing(client):
    client.size = 0
    assert read().status == "missing"


def test_last_trade_does_not_refresh_old_bid_ask(client):
    client.age = 60
    assert read(max_age=30).status == "stale"


def test_no_bid_timestamp_is_unknown(client):
    client.send_bid = False
    assert read().status == "unknown"


def test_crossed_quote(client):
    client.bid = 601
    assert read().status == "invalid"


@pytest.mark.parametrize("matches", [0, 2])
def test_ambiguous_or_unknown_contract(client, matches):
    client.details *= matches
    with pytest.raises(ibkr.PaperAccountError, match="matches"):
        read()
    assert "quote" not in client.calls
    assert client.disconnected


def test_wrong_account_stops_before_resolution(client):
    client.accounts = ["U999"]
    with pytest.raises(ibkr.PaperAccountError, match="not available"):
        read()
    assert client.calls == ["connect"]
    assert client.disconnected


def test_subscription_error_is_diagnostic_without_account_leak(client):
    client.error = True
    quote = read()
    assert quote.status == "incomplete"
    assert not quote.snapshot_complete
    assert "10089" in quote.diagnostics[0]
    assert "DU123" not in quote.diagnostics[0]
    assert client.disconnected


def test_timeout_keeps_partial_prices_but_not_success(client):
    client.hang = True
    quote = asyncio.run(
        ibkr.read_stock_quote(config(timeout=0.01), ibkr.StockQuoteRequest("SPY"))
    )
    assert quote.status == "incomplete"
    assert quote.bid == Decimal("600.25")
    assert client.disconnected


def test_cancel_disconnects_and_removes_handlers(client):
    async def cancel(*args, **kwargs):
        raise asyncio.CancelledError

    client.reqTickersAsync = cancel
    with pytest.raises(asyncio.CancelledError):
        read()
    assert client.disconnected
    assert client.pendingTickersEvent.handlers == client.errorEvent.handlers == []


def test_disconnect_during_read_fails(client):
    client.connected = False
    with pytest.raises(ibkr.PaperAccountError, match="disconnected"):
        read()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(symbol=""),
        dict(exchange=""),
        dict(currency=""),
        dict(market_data_type="invalid"),
        dict(max_age=0),
        dict(max_age=float("nan")),
    ],
)
def test_invalid_request(kwargs):
    with pytest.raises(ValueError):
        ibkr.StockQuoteRequest(**(dict(symbol="SPY") | kwargs))


@pytest.mark.parametrize("bad_quote,code", [(False, 0), (True, 2)])
def test_cli_uses_api_and_reports_data_status(
    client, tmp_path, monkeypatch, capsys, bad_quote, code
):
    import json
    from scripts import ibkr_quote

    monkeypatch.setenv("IBKR_ACCOUNT", "DU123")
    monkeypatch.setenv("IBKR_PORT", "7497")
    monkeypatch.setenv("IBKR_PAPER_CONFIRMED", "true")
    if bad_quote:
        client.bid = -1
    monkeypatch.setattr("sys.argv", ["ibkr_quote", "--symbol", "SPY"])
    assert ibkr_quote.main() == code
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == ("missing" if bad_quote else "received")
