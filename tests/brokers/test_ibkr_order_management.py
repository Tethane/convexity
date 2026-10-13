import asyncio
import copy
from dataclasses import replace
from decimal import Decimal as D
from itertools import count
from types import SimpleNamespace as NS

import pytest

from convexity.brokers import ibkr_trading
from convexity.brokers.ibkr import PaperAccountError
from convexity.trading import OrderJournal, StockOrder
from tests.brokers.test_ibkr import config
from tests.brokers.test_ibkr_trading import TradingClient


class ManagedClient(TradingClient):
    def __init__(self, journal):
        super().__init__(journal)
        self.client.getReqId = count(10).__next__
        self.client.cancelOrder = lambda order_id, _: self.cancelled.append(order_id)
        self.failure_at = None
        self.live = {}
        self.group_size = 0

    def placeOrder(self, stock, request):
        assert self.journal.get(request.orderRef).uncertain
        if self.group_size:
            assert len(self.journal.orders()) == self.group_size
        self.sent.append(copy.copy(request))
        if self.failure_at == len(self.sent):
            raise ConnectionError('transport interrupted')
        request.clientId = 71
        trade = NS(contract=stock, order=copy.copy(request), log=[],
                   orderStatus=NS(status='PendingSubmit', filled=0, permId=0))
        self.live[request.orderRef] = trade
        return trade

    def acknowledge(self, identity, *, status='Submitted', filled=0, terms=True):
        trade = self.live[identity]
        trade.order.permId = trade.order.orderId + 1000
        trade.orderStatus.status, trade.orderStatus.filled = status, filled
        (self.openOrderEvent if terms else self.orderStatusEvent).emit(trade)
        return trade


@pytest.fixture
def setup(tmp_path, monkeypatch):
    with OrderJournal(tmp_path / 'native.sqlite3') as journal:
        client = ManagedClient(journal)
        monkeypatch.setattr(ibkr_trading, '_new_client', lambda: client)
        yield journal, client


def stock(identity='order', kind='limit', side='BUY', quantity=10):
    return StockOrder(identity, 'owner', 'SPY', side, quantity,
                      None if kind == 'stop' else D(100), order_type=kind,
                      stop_price=D(99) if kind != 'limit' else None)


@pytest.mark.parametrize('kind,native', [('stop', 'STP'), ('stop_limit', 'STP LMT')])
def test_native_stop_types_keep_trigger_and_limit_distinct(setup, kind, native):
    journal, client = setup
    async def run():
        async with ibkr_trading.PaperBroker(config(), journal) as broker:
            await broker.submit(stock(kind=kind))
            order = client.sent[0]
            assert order.orderType == native and order.auxPrice == 99
            assert journal.get('order').request.stop_price == 99
            if kind == 'stop_limit':
                assert order.lmtPrice == 100
    asyncio.run(run())


def test_modify_same_id_requires_matching_open_order_and_preserves_fills(setup):
    journal, client = setup
    async def run():
        async with ibkr_trading.PaperBroker(config(), journal) as broker:
            original = stock()
            await broker.submit(original)
            client.acknowledge('order', filled=2)
            updated = replace(original, quantity=8, limit_price=D(101))
            pending = await broker.replace('order', updated, command_id='amend-1')
            assert pending.replace_pending and pending.request.quantity == 10
            assert pending.revision == 0 and pending.filled_quantity == 2
            assert client.sent[-1].orderId == client.sent[0].orderId
            client.acknowledge('order', filled=2, terms=False)
            assert journal.get('order').replace_pending  # status cannot confirm terms
            old = copy.copy(client.live['order'])
            old.order = copy.copy(client.sent[0])
            old.order.clientId = 71
            client.openOrderEvent.emit(old)
            assert journal.get('order').replace_pending  # stale terms cannot confirm
            client.acknowledge('order', filled=2)
            row = journal.get('order')
            assert row.request == updated and not row.replace_pending and not row.uncertain
            assert row.filled_quantity == 2 and row.remaining_quantity == 6
            await broker.replace('order', updated, command_id='amend-1')
            assert len(client.sent) == 2
    asyncio.run(run())


def test_modify_transport_failure_remains_pending_and_never_retries(setup):
    journal, client = setup
    async def run():
        async with ibkr_trading.PaperBroker(config(), journal) as broker:
            original = stock()
            await broker.submit(original)
            client.acknowledge('order')
            client.failure_at = 2
            updated = replace(original, limit_price=D(101))
            with pytest.raises(ConnectionError):
                await broker.replace('order', updated, command_id='amend-1')
            assert journal.get('order').replace_pending and journal.get('order').uncertain
            await broker.replace('order', updated, command_id='amend-1')
            assert len(client.sent) == 2
            with pytest.raises(PaperAccountError, match='unresolved'):
                await broker.submit(stock('new'))
    asyncio.run(run())


def test_modify_rejection_does_not_reject_the_working_order(setup):
    journal, client = setup
    async def run():
        async with ibkr_trading.PaperBroker(config(), journal) as broker:
            await broker.submit(stock())
            client.acknowledge('order')
            await broker.replace('order', replace(stock(), limit_price=D(101)), command_id='amend-1')
            client.errorEvent.emit(10, 201, 'Modification rejected', None)
            row = journal.get('order')
            assert not row.terminal and not row.replace_pending and row.uncertain
            assert row.request.limit_price == 100
    asyncio.run(run())


@pytest.mark.parametrize('kind', ['bracket', 'oco'])
def test_group_committed_before_any_transport_with_native_links(setup, kind):
    journal, client = setup
    requests = (stock('group:entry'), stock('group:stop', 'stop', 'SELL'), stock('group:profit', side='SELL')) if kind == 'bracket' else (stock('group:0'), stock('group:1'))
    client.group_size = len(requests)
    async def run():
        async with ibkr_trading.PaperBroker(config(), journal) as broker:
            await broker.submit_group('group', kind, requests)
            assert [order.transmit for order in client.sent] == [False] * (len(requests) - 1) + [True]
            if kind == 'bracket':
                assert all(order.parentId == client.sent[0].orderId for order in client.sent[1:])
            peers = client.sent[1:] if kind == 'bracket' else client.sent
            assert len({order.ocaGroup for order in peers}) == 1
            assert all(order.ocaGroup.startswith('cx-') and order.ocaType == 1 for order in peers)
            for request in requests:
                client.acknowledge(request.client_order_id)
            await broker.submit_group('group', kind, requests)
            assert len(client.sent) == len(requests)
            assert all(not row.uncertain for row in journal.orders())
    asyncio.run(run())


def test_staged_group_failure_never_resends_or_claims_all_members_working(setup):
    journal, client = setup
    requests = (stock('group:0'), stock('group:1'))
    client.failure_at = 2
    async def run():
        async with ibkr_trading.PaperBroker(config(), journal) as broker:
            with pytest.raises(ConnectionError):
                await broker.submit_group('group', 'oco', requests)
            client.acknowledge('group:0')
            assert journal.group('group')['transport_state'] == 'uncertain'
            assert all(row.uncertain for row in journal.orders())
            await broker.submit_group('group', 'oco', requests)
            assert len(client.sent) == 2
            with pytest.raises(PaperAccountError, match='group'):
                await broker.submit(stock('new'))
            with pytest.raises(ValueError, match='different'):
                await broker.submit_group('group', 'oco', (requests[0], replace(requests[1], quantity=12)))
    asyncio.run(run())


def test_cancel_transport_failure_never_fakes_terminal_state(setup):
    journal, client = setup
    def fail(*_):
        raise ConnectionError('cancel failed')
    async def run():
        async with ibkr_trading.PaperBroker(config(), journal) as broker:
            await broker.submit(stock())
            client.acknowledge('order')
            client.client.cancelOrder = fail
            with pytest.raises(ConnectionError):
                await broker.cancel('order')
            row = journal.get('order')
            assert row.cancel_pending and row.uncertain and not row.terminal
    asyncio.run(run())


def test_noop_modification_has_no_transport_or_accepted_revision(setup):
    journal, client = setup
    async def run():
        async with ibkr_trading.PaperBroker(config(), journal) as broker:
            await broker.submit(stock())
            client.acknowledge('order')
            result = await broker.replace('order', stock(), command_id='noop')
            assert result.revision == 0 and not result.replace_pending
            assert len(client.sent) == 1
    asyncio.run(run())


def test_pending_replacement_survives_reopen_then_reconciles_actual_terms(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from convexity.trading import Execution
    path = tmp_path / 'reopen.sqlite3'
    original = stock()
    desired = replace(original, quantity=9, limit_price=D(101))
    async def run():
        with OrderJournal(path) as journal:
            client = ManagedClient(journal)
            monkeypatch.setattr(ibkr_trading, '_new_client', lambda: client)
            async with ibkr_trading.PaperBroker(config(), journal) as broker:
                await broker.submit(original)
                journal.execution(Execution('prior-fill', 'order', D(2), D(100), 'BUY', datetime.now(timezone.utc).isoformat()))
                client.acknowledge('order', filled=2)
                await broker.replace('order', desired, command_id='rev1')
                replay = client.live['order']
                replay.order.permId = 1010
                replay.orderStatus.status, replay.orderStatus.filled = 'Submitted', 2
        with OrderJournal(path) as journal:
            before = journal.get('order')
            assert before.replace_pending and before.needs_terms and before.pending_request == desired
            client = ManagedClient(journal)
            async def opened():
                return [replay]
            client.reqOpenOrdersAsync = opened
            monkeypatch.setattr(ibkr_trading, '_new_client', lambda: client)
            async with ibkr_trading.PaperBroker(config(), journal):
                row = journal.get('order')
                assert row.request == desired and row.revision == 1
                assert not row.replace_pending and not row.needs_terms and not row.uncertain
                assert row.filled_quantity == 2 and row.remaining_quantity == 7
                assert row.broker_order_id == 10 and len(row.executions) == 1
                assert not client.sent
    asyncio.run(run())


def test_ambiguous_group_recovers_only_after_every_native_member_observed(setup):
    journal, client = setup
    requests = (stock('group:0'), stock('group:1'))
    client.failure_at = 2
    async def run():
        async with ibkr_trading.PaperBroker(config(), journal) as broker:
            with pytest.raises(ConnectionError):
                await broker.submit_group('group', 'oco', requests)
            client.acknowledge('group:0')
            assert journal.group('group')['transport_state'] == 'uncertain'
            order = copy.copy(client.sent[-1])
            order.clientId, order.permId = 71, 1011
            recovered = NS(contract=client.details[0].contract, order=order, log=[],
                           orderStatus=NS(status='Submitted', filled=0, permId=1011))
            client.openOrderEvent.emit(recovered)
            assert journal.group('group')['transport_state'] == 'sent'
            assert all(not row.uncertain for row in journal.orders())
            assert len(client.sent) == 2
    asyncio.run(run())


def test_completed_replacement_confirms_revised_terms_before_inferred_full_quantity(setup):
    journal, client = setup
    async def run():
        async with ibkr_trading.PaperBroker(config(), journal) as broker:
            await broker.submit(stock())
            client.acknowledge('order')
            desired = replace(stock(), quantity=8)
            await broker.replace('order', desired, command_id='rev1')
            completed = client.live['order']
            completed.orderStatus.status = 'Filled'
            completed.orderStatus.filled = 0  # native completedOrder omits cumulative quantity
            broker._on_completed_trade(completed)
            broker._check()
            row = journal.get('order')
            assert row.request.quantity == 8 and row.filled_quantity == 8 and row.revision == 1
            assert not row.replace_pending and row.terminal
            # Missing execution details still block fresh intents.
            with pytest.raises(PaperAccountError, match='missing execution'):
                await broker.submit(stock('new'))
    asyncio.run(run())


@pytest.mark.parametrize('field,value', [('symbol', 'QQQ'), ('currency', 'CAD')])
def test_strict_qualified_stock_identity_still_rejects_before_intent_or_transport(setup, field, value):
    journal, client = setup
    setattr(client.details[0].contract, field, value)
    guarded = []
    async def run():
        async with ibkr_trading.PaperBroker(config(), journal) as broker:
            with pytest.raises(PaperAccountError, match='identity differs'):
                await broker.submit(stock(), submission_guard=lambda: guarded.append(True))
            assert guarded == [True]
            assert not journal.orders() and not client.sent
    asyncio.run(run())
