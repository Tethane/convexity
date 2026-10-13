import asyncio
from dataclasses import replace
from decimal import Decimal as D

import pytest

from convexity.runtime import (BracketOrder, CancelOrders, OCOOrders, ReplaceOrder,
                               Rejection, Strategy, SubmitOrder)
from convexity.trading import OrderJournal, Execution
from tests.runtime.test_paper import FakeBroker, bridge, now


class ManagedBroker(FakeBroker):
    async def replace(self, identity, request, *, command_id, submission_guard=None):
        if submission_guard:
            submission_guard()
        if self.journal.replace_requested(identity, request, command_id):
            self.journal.replacement_observed(identity, request)
            self.on_update(self.journal.get(identity))
        return self.journal.get(identity)

    async def submit_group(self, group_id, kind, requests, *, submission_guard=None):
        if submission_guard:
            submission_guard()
        if self.journal.group_intent(group_id, kind, self.config.account, requests,
                                     tuple(range(len(self.sent) + 1, len(self.sent) + len(requests) + 1)), 71,
                                     (123,) * len(requests)):
            self.sent.extend(requests)
            self.journal.group_transport(group_id, 'sent')
            for request in requests:
                self.journal.status(request.client_order_id, 'Submitted', D(0))
                self.on_update(self.journal.get(request.client_order_id))
        return tuple(self.journal.get(request.client_order_id) for request in requests)


def test_stop_replacement_and_owner_scoped_bulk_cancel(tmp_path):
    async def run():
        with OrderJournal(tmp_path / 'journal.sqlite') as journal:
            broker = ManagedBroker(journal)
            async with bridge(broker, {'alpha': Strategy(), 'beta': Strategy()}) as live:
                await live.command('alpha', SubmitOrder.stop('A', 'BUY', 2, 105, client_order_id='entry'))
                await live.command('beta', SubmitOrder.limit('A', 'BUY', 1, 95, client_order_id='entry'))
                assert broker.sent[0].order_type == 'stop' and broker.sent[0].limit_price is None
                await live.command('alpha', ReplaceOrder('alpha:entry', quantity=3, stop_price=104, client_command_id='rev1'))
                order = live.context('alpha').orders[0]
                assert order.request.stop_price == 104 and order.request.quantity == 3
                assert order.revision == 1 and not order.replace_pending
                await live.command('alpha', CancelOrders(symbol='A'))
                assert live.context('alpha').orders[0].terminal
                assert not live.context('beta').orders[0].terminal
    asyncio.run(run())


def test_bracket_children_held_until_complete_parent_and_group_replay_idempotent(tmp_path):
    async def run():
        with OrderJournal(tmp_path / 'journal.sqlite') as journal:
            broker = ManagedBroker(journal)
            async with bridge(broker, {'s': Strategy()}) as live:
                command = BracketOrder(SubmitOrder.limit('A', 'BUY', 2, 100), D(110), D(90), 'bracket')
                result = await live.command('s', command)
                assert not any(isinstance(event, Rejection) for event in result)
                orders = live.context('s').orders
                assert {order.order_id for order in orders} == {'s:bracket:entry', 's:bracket:stop_loss', 's:bracket:take_profit'}
                parent = next(order for order in orders if order.parent_order_id is None)
                children = [order for order in orders if order.parent_order_id]
                assert all(not child.active and child.group_id == 's:bracket' for child in children)
                broker.fill(1, commission=D(0))
                assert all(not order.active for order in live.context('s').orders if order.parent_order_id)
                local_id = journal.broker_order_id('DU123', parent.order_id)
                journal.execution(Execution('fill-2', local_id, D(1), D(100), 'BUY', now().isoformat(), D(0), 'USD'))
                assert all(order.active for order in live.context('s').orders if order.parent_order_id)
                await live.command('s', command)
                assert len(broker.sent) == 3
                await live.command('s', CancelOrders(group_id='bracket'))
                assert all(order.terminal for order in live.context('s').orders)
    asyncio.run(run())


def test_group_collision_does_not_send_or_change_prior_orders(tmp_path):
    async def run():
        with OrderJournal(tmp_path / 'journal.sqlite') as journal:
            broker = ManagedBroker(journal)
            async with bridge(broker, {'s': Strategy()}) as live:
                command = OCOOrders((SubmitOrder.limit('A', 'BUY', 1, 95), SubmitOrder.stop('A', 'BUY', 1, 105)), 'choice')
                await live.command('s', command)
                before = journal.orders()
                conflict = replace(command, orders=(replace(command.orders[0], quantity=2), command.orders[1]))
                result = await live.command('s', conflict)
                assert any(isinstance(event, Rejection) for event in result)
                assert len(broker.sent) == 2 and journal.orders() == before
    asyncio.run(run())


def test_replacement_cannot_mutate_another_owner(tmp_path):
    async def run():
        with OrderJournal(tmp_path / 'journal.sqlite') as journal:
            broker = ManagedBroker(journal)
            async with bridge(broker, {'alpha': Strategy(), 'beta': Strategy()}) as live:
                await live.command('alpha', SubmitOrder.limit('A', 'BUY', 1, 95, client_order_id='entry'))
                result = await live.command('beta', ReplaceOrder('alpha:entry', limit_price=94, client_command_id='rev'))
                assert any(isinstance(event, Rejection) for event in result)
                assert journal.orders()[0].revision == 0
    asyncio.run(run())


def test_oco_replacement_reserves_one_alternative_and_group_replay_uses_original(tmp_path):
    async def run():
        with OrderJournal(tmp_path / 'journal.sqlite') as journal:
            broker = ManagedBroker(journal)
            async with bridge(broker, {'s': Strategy()}) as live:
                command = OCOOrders((SubmitOrder.limit('A', 'BUY', 9, 100), SubmitOrder.limit('A', 'BUY', 9, 99)), 'choice')
                await live.command('s', command)
                result = await live.command('s', ReplaceOrder('s:choice:0', limit_price=99, client_command_id='rev'))
                assert not any(isinstance(event, Rejection) for event in result)
                assert live.context('s').orders[0].revision == 1
                replay = await live.command('s', command)
                assert not any(isinstance(event, Rejection) for event in replay)
                assert len(broker.sent) == 2
    asyncio.run(run())


def test_successive_operator_reprices_without_ids_are_independent_intents(tmp_path):
    async def run():
        with OrderJournal(tmp_path / 'journal.sqlite') as journal:
            broker = ManagedBroker(journal)
            async with bridge(broker, {'s': Strategy()}) as live:
                await live.command('s', SubmitOrder.limit('A', 'BUY', 1, 95, client_order_id='entry'))
                for price in (96, 97):
                    result = await live.command('s', ReplaceOrder('s:entry', limit_price=price))
                    assert not any(isinstance(event, Rejection) for event in result)
                assert live.context('s').orders[0].request.limit_price == 97
                assert live.context('s').orders[0].revision == 2
    asyncio.run(run())
