"""Declarations govern observations and commands identically across runtimes."""
import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from convexity.backtest import SimulatedExchange
from convexity.runtime import (Bar, Bars, BarSubscription, ClockSubscription, ContractViolation,
    EventSubscription, FillEvent, Instrument, OrderUpdate, Runtime, Strategy,
    StrategyContract, StrategyMetadata, SubmitOrder, Timer)
from convexity.trading import OrderJournal
from tests.runtime.test_paper import FakeBroker, bridge

T = datetime(2026, 1, 2, 15, tzinfo=timezone.utc)
H = timedelta(hours=1)


def panel(time, *symbols):
    return Bars(time, {symbol: Bar(symbol, time-H, time, Decimal(100), Decimal(100), Decimal(100), Decimal(100), 100) for symbol in symbols})


def simulation(strategies):
    instruments = {s: Instrument(s) for s in ('A', 'B')}
    result = Runtime(strategies, SimulatedExchange(instruments), instruments)
    result.marks.update(A=Decimal(100), B=Decimal(100))
    return result


def test_contract_violation_validates_entire_batch_before_submitting():
    class Invalid(Strategy):
        contract = StrategyContract(('A',))
        def on_start(self, context):
            yield SubmitOrder('A', 'BUY', 1)
            yield SubmitOrder('B', 'BUY', 1)
    runtime = simulation({'invalid': Invalid()})
    with pytest.raises(ContractViolation, match="'B'.*outside"):
        runtime.start(T)
    assert not runtime.orders
    assert runtime.paused == {'invalid'}


def test_subscriptions_filter_panels_context_and_named_clocks():
    class Observer(Strategy):
        contract = StrategyContract(('A',), (BarSubscription(('A',)), ClockSubscription('close')))
        def __init__(self):
            self.events = []
        def on_bar(self, event, context):
            self.events.append(event)
            assert set(context.marks) == {'A'}
            assert set(context.history) == {'A'}
        def on_timer(self, event, context):
            self.events.append(event)
        def on_event(self, event, context):
            raise AssertionError('A named callback must be dispatched only once')
    observer = Observer()
    runtime = simulation({'observer': observer})
    runtime.start(T)
    runtime.advance(T+H, panel(T+H, 'A', 'B'), (Timer(T+H, 'close'), Timer(T+H, 'other')))
    assert len(observer.events) == 2
    assert set(observer.events[0].bars) == {'A'}


def test_owner_warmup_and_live_observations_do_not_leak():
    class Observer(Strategy):
        contract = StrategyContract(('A',), (BarSubscription(('A',)),), warmup_sessions=1)
    runtime = simulation({'warm': Observer(), 'cold': Observer()})
    runtime.history['A'] = [panel(T, 'A').bars['A']]
    runtime.warmup_history('warm', runtime.history['A'])
    runtime.start(T)
    assert len(runtime.context('warm').history['A']) == 1
    assert not runtime.context('cold').history['A']
    runtime.advance(T+H, panel(T+H, 'A'), publish_bars=False,
                    observations_by_owner={'cold': panel(T+H, 'A')})
    assert len(runtime.context('warm').history['A']) == 1
    assert runtime.context('cold').history['A'][0].end == T+H


def test_timer_scope_prevents_same_name_cross_delivery():
    class Observer(Strategy):
        contract = StrategyContract((), (ClockSubscription('tick'),))
        def __init__(self):
            self.calls = 0
        def on_timer(self, event, context):
            self.calls += 1
    a, b = Observer(), Observer()
    runtime = simulation({'a': a, 'b': b})
    runtime.start(T)
    timer = Timer(T+H, 'tick')
    runtime.advance(timer.time, timers=(timer,), timer_owners={timer: ('a',)})
    assert (a.calls, b.calls) == (1, 0)


def test_paper_batch_contract_violation_reaches_no_broker(tmp_path):
    class Invalid(Strategy):
        contract = StrategyContract(('A',))
        def on_start(self, context):
            return [SubmitOrder('A', 'BUY', 1), SubmitOrder('B', 'BUY', 1)]
    async def run():
        with OrderJournal(tmp_path / 'orders.sqlite') as journal:
            broker = FakeBroker(journal)
            async with bridge(broker, {'invalid': Invalid()}) as runtime:
                assert not broker.sent
                assert 'invalid' in runtime.paused
                assert any('ContractViolation' in error for error in runtime.errors)
    asyncio.run(run())


def test_paper_receives_canonical_partial_and_fill_callbacks(tmp_path):
    class Buyer(Strategy):
        contract = StrategyContract(('A',), (EventSubscription(OrderUpdate), EventSubscription(FillEvent)))
        def __init__(self):
            self.partials, self.fills = [], []
        def on_start(self, context):
            return SubmitOrder.market('A', 'BUY', 2)
        def on_partial(self, event, context):
            self.partials.append(event)
        def on_fill(self, event, context):
            self.fills.append(event)
    async def run():
        with OrderJournal(tmp_path / 'orders.sqlite') as journal:
            broker, buyer = FakeBroker(journal), Buyer()
            async with bridge(broker, {'buyer': buyer}) as runtime:
                broker.fill(1, commission=Decimal('0'))
                await runtime.refresh()
                assert len(buyer.partials) == len(buyer.fills) == 1
                assert isinstance(buyer.partials[0], OrderUpdate)
                assert isinstance(buyer.fills[0], FillEvent)
    asyncio.run(run())


def test_named_order_constructors_are_explicit_and_validate():
    assert SubmitOrder.market('A', 'BUY', 2).time_in_force == 'DAY'
    order = SubmitOrder.limit('A', 'BUY', 2, '99.50', time_in_force='FOK')
    assert order.order_type == 'limit' and order.limit_price == Decimal('99.50')
    assert order.time_in_force == 'FOK'
