import asyncio
from decimal import Decimal
from types import SimpleNamespace as NS

import pytest

from convexity.brokers import ibkr_trading
from convexity.brokers.ibkr import PaperAccountError
from convexity.trading import OrderJournal
from tests.brokers.test_ibkr import FakeClient, config, contract, pytestmark
from tests.brokers.test_ibkr_quote import Event
from tests.trading.test_journal import order


class TradingClient(FakeClient):
    openOrderEvent: Event
    orderStatusEvent: Event

    def __init__(self, journal):
        super().__init__()
        self.journal = journal
        for name in (
            "openOrderEvent",
            "orderStatusEvent",
            "execDetailsEvent",
            "commissionReportEvent",
            "errorEvent",
            "disconnectedEvent",
        ):
            setattr(self, name, Event())
        self.client = NS(getReqId=lambda: 7, cancelOrder=self.cancel_transport)
        self.sent = []
        self.cancelled = []
        self.raise_on_send = False
        self.expected_tif = "DAY"
        self.expected_exchange = "OVERNIGHT"
        self.details = [
            NS(contract=contract(), validExchanges="SMART,OVERNIGHT", minTick=0.01)
        ]

    async def reqOpenOrdersAsync(self):
        return []

    async def reqCompletedOrdersAsync(self, apiOnly):
        return []

    async def reqExecutionsAsync(self, executionFilter):
        return []

    async def reqContractDetailsAsync(self, stock):
        return self.details

    def placeOrder(self, stock, request):
        assert self.journal.get(
            request.orderRef
        ).uncertain  # committed before transport
        assert request.account == "DU123"
        assert request.tif == self.expected_tif
        assert not request.outsideRth  # Overnight route has its own session.
        assert stock.exchange == self.expected_exchange
        assert request.lmtPrice == 700.01
        self.sent.append(request)
        if self.raise_on_send:
            raise ConnectionError("uncertain transport")
        request.clientId = 71
        trade = NS(
            order=request, orderStatus=NS(status="PendingSubmit", filled=0, permId=0)
        )
        return trade

    def cancel_transport(self, broker_id, manual_time):
        assert self.journal.get("order-1").cancel_pending
        self.cancelled.append(broker_id)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    with OrderJournal(tmp_path / "orders.sqlite") as journal:
        client = TradingClient(journal)
        monkeypatch.setattr(ibkr_trading, "_new_client", lambda: client)
        yield journal, client


def test_owned_submission_ack_cancel_is_broker_confirmed(setup):
    journal, client = setup

    async def run():
        async with ibkr_trading.PaperBroker(config(), journal) as broker:
            pending = await broker.submit(order())
            assert pending.uncertain and not pending.acknowledged
            again = await broker.submit(order())
            assert again.broker_order_id == pending.broker_order_id
            assert len(client.sent) == 1
            t = broker._trades["order-1"]
            t.order.permId = 555
            t.orderStatus.status = "Submitted"
            client.orderStatusEvent.emit(t)
            row = await broker.observe("order-1", until="acknowledged", seconds=0.1)
            assert row.acknowledged and not row.uncertain
            await broker.cancel("order-1")
            row = await broker.observe("order-1", seconds=0.01)
            assert row.cancel_pending and not row.terminal
            assert client.cancelled == [7]
            t.orderStatus.status = "Cancelled"
            client.orderStatusEvent.emit(t)
            row = await broker.observe("order-1", seconds=0.1)
            assert row.terminal and not row.cancel_pending

    asyncio.run(run())
    assert client.disconnected
    assert client.errorEvent.handlers == []


def test_transport_failure_persists_and_never_retries(setup):
    journal, client = setup

    async def run():
        async with ibkr_trading.PaperBroker(config(), journal) as broker:
            client.raise_on_send = True
            with pytest.raises(ConnectionError):
                await broker.submit(order())
            assert journal.get("order-1").uncertain
            await broker.submit(order())
            assert len(client.sent) == 1
            with pytest.raises(PaperAccountError, match="unresolved"):
                await broker.submit(order("different"))

    asyncio.run(run())


def test_wrong_account_never_sends(setup):
    journal, client = setup
    client.accounts = ["U999"]

    async def run():
        with pytest.raises(PaperAccountError, match="not available"):
            async with ibkr_trading.PaperBroker(config(), journal) as broker:
                await broker.submit(order())

    asyncio.run(run())
    assert client.sent == []
    assert journal.orders() == ()
    assert client.disconnected


def test_submission_timeout_and_disconnect_stay_unresolved(setup):
    journal, client = setup

    async def run():
        async with ibkr_trading.PaperBroker(config(), journal) as broker:
            await broker.submit(order())
            row = await broker.observe("order-1", seconds=0.01)
            assert row.uncertain and not row.terminal
            client.connected = False
            client.disconnectedEvent.emit()
            with pytest.raises(PaperAccountError, match="not connected"):
                await broker.submit(order("second"))

    asyncio.run(run())
    assert journal.get("order-1").uncertain


def test_broker_rejection_is_visible(setup):
    journal, client = setup

    async def run():
        async with ibkr_trading.PaperBroker(config(), journal) as broker:
            await broker.submit(order())
            t = broker._trades["order-1"]
            t.orderStatus.status = "Cancelled"
            client.orderStatusEvent.emit(t)
            client.errorEvent.emit(7, 201, "Order rejected for DU123", contract())
            result = await broker.observe("order-1", seconds=0.1)
            assert result.status == "Rejected"
            assert "DU123" not in result.diagnostics[0]

    asyncio.run(run())


def test_execution_replay_and_attribution(setup):
    from datetime import datetime, timezone

    journal, client = setup

    async def run():
        async with ibkr_trading.PaperBroker(config(), journal) as broker:
            await broker.submit(order())
            execution = NS(
                execId="one",
                acctNumber="DU123",
                clientId=71,
                orderId=7,
                permId=55,
                orderRef="order-1",
                side="BOT",
                shares=1,
                price=700,
                time=datetime.now(timezone.utc),
            )
            fill = NS(contract=contract(), execution=execution)
            client.execDetailsEvent.emit(None, fill)
            client.execDetailsEvent.emit(None, fill)
            assert len(journal.get("order-1").executions) == 1
            assert journal.attributed_position("DU123", "smoke", 123) == 1
            execution.acctNumber = "DU999"
            execution.execId = "unowned"
            client.execDetailsEvent.emit(None, fill)
            assert len(journal.get("order-1").executions) == 1

    asyncio.run(run())


def test_recording_failure_blocks_further_commands(setup):
    journal, client = setup

    def fail(row):
        raise OSError("disk/log failure")

    async def run():
        async with ibkr_trading.PaperBroker(config(), journal) as broker:
            await broker.submit(order())
            broker.on_update = fail
            t = broker._trades["order-1"]
            t.orderStatus.status = "Submitted"
            client.orderStatusEvent.emit(t)
            with pytest.raises(PaperAccountError, match="recording failed"):
                await broker.submit(order("other"))

    asyncio.run(run())


def test_synthetic_library_cancellation_does_not_confirm_cancel(setup):
    journal, client = setup

    async def run():
        async with ibkr_trading.PaperBroker(config(), journal) as broker:
            await broker.submit(order())
            t = broker._trades["order-1"]
            t.orderStatus.status = "Cancelled"
            t.log = [NS(errorCode=10147)]
            client.orderStatusEvent.emit(t)
            client.errorEvent.emit(7, 10147, "Cannot find order to cancel", None)
            row = journal.get("order-1")
            assert not row.terminal
            assert row.uncertain
            client.errorEvent.emit(7, 202, "Order cancelled", None)
            assert journal.get("order-1").status == "Cancelled"

    asyncio.run(run())


def test_restart_unknown_id_is_not_resent(setup):
    journal, client = setup

    async def run():
        async with ibkr_trading.PaperBroker(config(), journal) as broker:
            await broker.submit(order())
        client.disconnected = False
        async with ibkr_trading.PaperBroker(config(), journal) as broker:
            row = await broker.submit(order())
            assert row.uncertain
            assert len(client.sent) == 1

    asyncio.run(run())


def test_cannot_cancel_another_client_order(setup):
    journal, client = setup
    journal.intent(order(), "DU123", 7, 99, 123)

    async def run():
        async with ibkr_trading.PaperBroker(config(), journal) as broker:
            with pytest.raises(PaperAccountError, match="another account/client"):
                await broker.cancel("order-1")

    asyncio.run(run())
    assert client.cancelled == []


def test_restart_with_terminal_fill_missing_execution_details_blocks_new_orders(setup):
    from tests.trading.test_journal import fill

    journal, client = setup
    journal.intent(order(), "DU123", 6, 71, 123)
    journal.status("order-1", "Filled", Decimal(2))

    async def run():
        async with ibkr_trading.PaperBroker(config(), journal) as broker:
            record = journal.get("order-1")
            assert record.terminal and record.uncertain
            assert "missing" in record.diagnostics[-1]
            with pytest.raises(PaperAccountError, match="missing execution"):
                await broker.submit(order("new-order"))
            assert not client.sent
            # Exact recovered executions resolve the gap without an invented fill.
            journal.execution(fill(qty="2"))
            await broker.submit(order("new-order"))
            assert len(client.sent) == 1

    asyncio.run(run())


def test_fill_status_arriving_during_contract_resolution_blocks_transport(setup):
    journal, client = setup
    journal.intent(order(), "DU123", 6, 71, 123)
    journal.status("order-1", "Submitted", Decimal(0))
    original_lookup = client.reqContractDetailsAsync

    async def lookup(stock):
        details = await original_lookup(stock)
        journal.status("order-1", "Filled", Decimal(2))
        return details

    client.reqContractDetailsAsync = lookup

    async def replay_open():
        journal.status("order-1", "Submitted", Decimal(0))
        return []

    client.reqOpenOrdersAsync = replay_open

    async def run():
        async with ibkr_trading.PaperBroker(config(), journal) as broker:
            with pytest.raises(PaperAccountError, match="missing execution"):
                await broker.submit(order("new-order"))
            assert not client.sent
            assert len(journal.orders()) == 1

    asyncio.run(run())


@pytest.mark.parametrize("replayed", [True, False])
def test_unknown_completed_filled_quantity_recovers_before_submission_gate(
    setup, replayed
):
    from datetime import datetime, timezone

    journal, client = setup
    journal.intent(order(), "DU123", 7, 71, 123)
    trade = NS(
        order=NS(
            account="DU123", clientId=71, orderId=7, permId=55, orderRef="order-1"
        ),
        orderStatus=NS(status="Filled", filled=0, permId=55),
        log=[],
    )

    async def completed(apiOnly):
        return [trade]

    async def executions(executionFilter):
        return (
            [
                NS(
                    contract=contract(),
                    execution=NS(
                        execId="recovered",
                        acctNumber="DU123",
                        clientId=71,
                        orderId=7,
                        permId=55,
                        orderRef="order-1",
                        side="BOT",
                        shares=2,
                        price=700,
                        time=datetime.now(timezone.utc),
                    ),
                )
            ]
            if replayed
            else []
        )

    client.reqCompletedOrdersAsync = completed
    client.reqExecutionsAsync = executions

    async def run():
        async with ibkr_trading.PaperBroker(config(), journal) as broker:
            row = journal.get("order-1")
            assert row.status == "Filled" and row.filled_quantity == 2
            assert row.executed_quantity == (2 if replayed else 0)
            if replayed:
                assert not row.uncertain
                await broker.submit(order("new-intent"))
                assert len(client.sent) == 1
            else:
                assert row.uncertain and not row.executions
                with pytest.raises(PaperAccountError, match="missing execution"):
                    await broker.submit(order("new-intent"))
                assert not client.sent

    asyncio.run(run())


def test_live_filled_zero_does_not_infer_missing_quantity(setup):
    journal, client = setup

    async def run():
        async with ibkr_trading.PaperBroker(config(), journal) as broker:
            await broker.submit(order())
            trade = broker._trades["order-1"]
            trade.orderStatus.status = "Filled"
            trade.orderStatus.filled = 0
            broker._on_trade(trade)
            with pytest.raises(PaperAccountError, match="recording failed"):
                broker._check()
            assert journal.get("order-1").filled_quantity == 0

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["denied", "timeout"])
def test_connected_quote_data_failure_returns_qualified_unavailable_quote(
    tmp_path, failure
):
    from ib_async import RequestError
    from convexity.brokers.ibkr import StockQuoteRequest
    from tests.brokers.test_ibkr_quote import QuoteClient

    with OrderJournal(tmp_path / "quotes.sqlite") as journal:
        client = QuoteClient()
        broker = ibkr_trading.PaperBroker(config(timeout=0.02), journal)
        broker._ib = client
        if failure == "denied":

            async def denied(*args, **kwargs):
                raise RequestError(8, 10197, "Competing market-data session for DU123")

            client.reqTickersAsync = denied
        else:
            client.hang = True
        quote = asyncio.run(broker.quote(StockQuoteRequest("SPY")))
        assert quote.contract.contract_id == 123
        assert quote.status == "incomplete" and not quote.snapshot_complete
        assert quote.diagnostics and "DU123" not in str(quote.diagnostics)
        assert not client.pendingTickersEvent.handlers
        assert client.connected


@pytest.mark.parametrize("failure", ["identity", "disconnect"])
def test_connected_quote_does_not_hide_contract_or_connection_failure(
    tmp_path, failure
):
    from ib_async import RequestError
    from convexity.brokers.ibkr import StockQuoteRequest
    from tests.brokers.test_ibkr_quote import QuoteClient

    with OrderJournal(tmp_path / "quotes.sqlite") as journal:
        client = QuoteClient()
        broker = ibkr_trading.PaperBroker(config(), journal)
        broker._ib = client
        if failure == "identity":
            client.details[0].contract.symbol = "WRONG"
        else:

            async def disconnected(*args, **kwargs):
                client.connected = False
                raise RequestError(8, 10197, "session unavailable")

            client.reqTickersAsync = disconnected
        with pytest.raises(PaperAccountError):
            asyncio.run(broker.quote(StockQuoteRequest("SPY")))
        assert not client.pendingTickersEvent.handlers


def test_gtc_transmitted_and_conflicting_tif_never_resubmits(setup):
    from dataclasses import replace

    journal, client = setup
    client.expected_tif = "GTC"
    client.expected_exchange = "SMART"
    request = replace(order(), session="regular", time_in_force="GTC")

    async def run():
        async with ibkr_trading.PaperBroker(config(), journal) as broker:
            await broker.submit(request)
            assert client.sent[0].tif == "GTC"
            with pytest.raises(ValueError, match="different intent"):
                await broker.submit(replace(request, time_in_force="DAY"))
            assert len(client.sent) == 1

    asyncio.run(run())
