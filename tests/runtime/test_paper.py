"""Paper bridge behavior with real durable journal and deterministic broker doubles."""
import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
from types import SimpleNamespace
from typing import Callable, cast
from convexity.brokers.ibkr_trading import PaperBroker
from convexity.trading import OrderRecord

import pytest

from convexity.brokers.ibkr import AccountValue, Contract, PaperAccountSnapshot, Position as BrokerPosition, Quote
from convexity.runtime import FillEvent, Instrument, Rejection, Strategy, SubmitOrder, Timer
from convexity.runtime.paper import PaperRuntime
from convexity.runtime.state import DeliveryError
from convexity.trading import Execution, OrderJournal, StockOrder


def now():
    return datetime.now(timezone.utc)


STOCK = Contract(123, "A", "STK", "A", "USD", "SMART", "", D(0), "", "", ())
BOND = replace(STOCK, contract_id=999, symbol="BOND", security_type="BOND")


class FakeBroker:
    def __init__(self, journal, *, funds=1000, manual=0):
        self.journal = journal
        self.config = SimpleNamespace(account="DU123")
        self.on_update: Callable[[OrderRecord], None] | None = None
        self.funds, self.manual = funds, manual
        self.sent = []
        self.stale = False
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True

    async def quote(self, request):
        stamp = now() - timedelta(hours=1) if self.stale else now()
        return Quote(STOCK, "live", "live", D(100), D(100), D(10), D(10), stamp, stamp, now(), 30, True, "received", ())

    async def snapshot(self):
        quantity = D(self.manual) + sum((execution.quantity if execution.side == "BUY" else -execution.quantity
                                       for record in self.journal.orders() if isinstance(record.request, StockOrder)
                                       for execution in record.executions), D(0))
        return PaperAccountSnapshot("DU123", now(), now(),
            (AccountValue("NetLiquidation", "20000", "USD"), AccountValue("TotalCashValue", "10000", "USD"),
             AccountValue("AvailableFunds", str(self.funds), "USD")),
            (BrokerPosition(STOCK, quantity, D(100)), BrokerPosition(BOND, D(1000), D(99))), ())

    async def submit(self, request, *, submission_guard=None):
        if submission_guard is not None:
            submission_guard()
        self.sent.append(request)
        self.journal.intent(request, "DU123", len(self.sent), 71, 123)
        self.journal.status(request.client_order_id, "Submitted", D(0))
        record = self.journal.get(request.client_order_id)
        assert self.on_update is not None
        self.on_update(record)
        return record

    async def cancel(self, identity):
        record = self.journal.get(identity)
        self.journal.status(identity, "Cancelled", record.filled_quantity)
        record = self.journal.get(identity)
        assert self.on_update is not None
        self.on_update(record)
        return record

    def fill(self, quantity, *, commission=None):
        request = self.sent[0]
        self.journal.execution(Execution("execution-1", request.client_order_id, D(quantity), D(100), "BUY", now().isoformat(),
                                         commission, "USD" if commission is not None else None))
        assert self.on_update is not None
        self.on_update(self.journal.get(request.client_order_id))


class BuyOnTimer(Strategy):
    def __init__(self, quantity=1):
        self.quantity = quantity
        self.events = []

    def on_event(self, event, context):
        self.events.append(event)
        if isinstance(event, Timer):
            return SubmitOrder("A", "BUY", self.quantity)


def bridge(broker, strategies, **kwargs):
    return PaperRuntime(cast(PaperBroker, broker), strategies, {"A": Instrument("A")},
                        budgets={owner: D(1000) for owner in strategies}, max_order_notional=D(1000),
                        market_collar_bps=D(10), **kwargs)


def test_same_strategy_commands_become_collared_limits_and_restart_never_duplicates(tmp_path):
    async def run():
        with OrderJournal(tmp_path / "orders.sqlite") as journal:
            timer = Timer(now(), "rebalance")
            first = FakeBroker(journal)
            async with bridge(first, {"s": BuyOnTimer()}) as live:
                await live.step(timer)
                await live.step(timer)
                assert len(first.sent) == 1
                assert first.sent[0].limit_price == D("100.10")
                assert live.context("s").pending("A") == 1
                assert live.context("s").equity == 20000
            assert first.closed
            second = FakeBroker(journal)
            async with bridge(second, {"s": BuyOnTimer()}) as live:
                await live.step(timer)
                assert not second.sent
    asyncio.run(run())


def test_strategies_cannot_reserve_same_available_funds_twice(tmp_path):
    async def run():
        with OrderJournal(tmp_path / "orders.sqlite") as journal:
            broker = FakeBroker(journal, funds=1000)
            alpha, beta = BuyOnTimer(6), BuyOnTimer(6)
            async with bridge(broker, {"alpha": alpha, "beta": beta}) as live:
                await live.step(Timer(now(), "rebalance"))
                assert len(broker.sent) == 1
                assert any(isinstance(event, Rejection) and "AvailableFunds" in event.reason for event in beta.events)
    asyncio.run(run())


@pytest.mark.parametrize("stale,manual", [(True, 0), (False, 1)])
def test_stale_quotes_or_manual_configured_positions_block_submission(tmp_path, stale, manual):
    async def run():
        with OrderJournal(tmp_path / "orders.sqlite") as journal:
            broker = FakeBroker(journal, manual=manual)
            broker.stale = stale
            async with bridge(broker, {"s": BuyOnTimer()}) as live:
                await live.step(Timer(now(), "rebalance"))
                assert not broker.sent
                assert any(isinstance(event, Rejection) for event in live.events)
    asyncio.run(run())


def test_fills_update_ownership_once_but_never_simulate_broker_cash(tmp_path):
    async def run():
        with OrderJournal(tmp_path / "orders.sqlite") as journal:
            broker, strategy = FakeBroker(journal), BuyOnTimer(2)
            async with bridge(broker, {"s": strategy}) as live:
                await live.step(Timer(now(), "rebalance"))
                broker.fill(1)
                await live.refresh()
                assert live.context("s").position("A") == 1
                assert live.context("s").cash == 10000  # authoritative snapshot, not 9900
                assert not any(isinstance(event, FillEvent) for event in strategy.events)
                journal.commission("DU123", "execution-1", D(".35"), "USD")
                assert broker.on_update is not None
                broker.on_update(journal.get(broker.sent[0].client_order_id))
                await live.refresh()
                assert broker.on_update is not None
                broker.on_update(journal.get(broker.sent[0].client_order_id))
                await live.refresh()
                fills = [event for event in strategy.events if isinstance(event, FillEvent)]
                assert len(fills) == 1 and fills[0].fill.fee == D(".35")
    asyncio.run(run())


def test_failure_pauses_callbacks_without_losing_fill_tracking(tmp_path):
    class Broken(BuyOnTimer):
        def on_event(self, event, context):
            if isinstance(event, Timer) and event.name == "fail":
                raise RuntimeError("failed signal")
            return super().on_event(event, context)

    async def run():
        with OrderJournal(tmp_path / "orders.sqlite") as journal:
            broker = FakeBroker(journal)
            async with bridge(broker, {"s": Broken()}) as live:
                await live.step(Timer(now(), "rebalance"))
                with pytest.raises(DeliveryError, match="failed signal"):
                    await live.step(Timer(now(), "fail"))
                broker.fill(1, commission=D(".35"))
                await live.refresh()
                assert "s" in live.paused
                assert live.context("s").position("A") == 1
                assert len(broker.sent) == 1
    asyncio.run(run())


def test_missing_execution_in_unrelated_strategy_blocks_new_entries(tmp_path):
    from convexity.trading import StockOrder

    async def run():
        with OrderJournal(tmp_path / "orders.sqlite") as journal:
            journal.intent(StockOrder("older", "other", "B", "BUY", 1, D(100)), "DU123", 99, 71, 456)
            journal.status("older", "Filled", D(1))
            broker = FakeBroker(journal)
            async with bridge(broker, {"s": BuyOnTimer()}) as live:
                await live.step(Timer(now(), "rebalance"))
                assert not broker.sent
                assert any("Unresolved" in error for error in live.errors)
    asyncio.run(run())


def test_confirmed_owned_exit_is_allowed_with_exhausted_available_funds(tmp_path):
    class CloseOnTimer(BuyOnTimer):
        def on_event(self, event, context):
            if isinstance(event, Timer) and event.name == "exit":
                return SubmitOrder("A", "SELL", context.position("A"))
            return super().on_event(event, context)

    async def run():
        with OrderJournal(tmp_path / "orders.sqlite") as journal:
            broker = FakeBroker(journal)
            async with bridge(broker, {"s": CloseOnTimer()}) as live:
                await live.step(Timer(now(), "entry"))
                broker.fill(1, commission=D(".35"))
                await live.refresh()
                broker.funds = -100
                await live.step(Timer(now(), "exit"))
                assert len(broker.sent) == 2
                assert broker.sent[-1].side == "SELL"
    asyncio.run(run())


def test_vertical_commands_and_leg_updates_share_runtime_without_stock_fill_accounting(tmp_path):
    from convexity.brokers.ibkr_verticals import VerticalBroker
    from convexity.runtime import BrokerOrderUpdate, CancelOrder, OptionQuotes
    from convexity.trading import StockOrder
    from tests.trading.test_verticals import vertical

    class ComboFake(FakeBroker, VerticalBroker):
        def __init__(self, journal):
            FakeBroker.__init__(self, journal)
            self.strategy_budgets = {"spread": D(1000)}

        async def submit(self, request, *, submission_guard=None):
            if submission_guard is not None:
                submission_guard()
            self.sent.append(request)
            self.journal.intent(request, "DU123", len(self.sent), 71, 123 if isinstance(request, StockOrder) else 0)
            self.journal.status(request.client_order_id, "Submitted", D(0))
            record = self.journal.get(request.client_order_id)
            assert self.on_update is not None
            self.on_update(record)
            return record

    class Spread(Strategy):
        def __init__(self):
            self.events = []

        def on_event(self, event, context):
            self.events.append(event)
            if isinstance(event, OptionQuotes):
                return vertical()
            if isinstance(event, Timer) and event.name == "cancel":
                return CancelOrder("v1")

    async def run():
        with OrderJournal(tmp_path / "orders.sqlite") as journal:
            broker, strategy = ComboFake(journal), Spread()
            async with bridge(broker, {"spread": strategy}) as live:
                quote = await broker.quote(None)
                option = replace(quote, contract=replace(STOCK, contract_id=101, symbol="SPY", security_type="OPT"))
                await live.step(OptionQuotes(now(), (option,)))
                assert len(broker.sent) == 1
                assert len(live.context("spread").broker_orders) == 1
                for execution_id, contract_id, side in (("leg1", 101, "BUY"), ("leg2", 102, "SELL")):
                    journal.execution(Execution(execution_id, "v1", D(1), D(1), side, now().isoformat(), contract_id=contract_id))
                    assert broker.on_update is not None
                    broker.on_update(journal.get("v1"))
                    await live.refresh()
                assert live.context("spread").broker_orders[0].executed_quantity == 1
                updates = [event for event in strategy.events if isinstance(event, BrokerOrderUpdate)]
                assert [len(event.record.executions) for event in updates] == [0, 1, 2]
                assert not any(isinstance(event, FillEvent) for event in strategy.events)
                assert live.context("spread").positions == {}
                assert live.context("spread").cash == 10000
                assert broker.on_update is not None
                broker.on_update(journal.get("v1"))
                await live.refresh()
                assert len([event for event in strategy.events if isinstance(event, BrokerOrderUpdate)]) == 3
    asyncio.run(run())


def test_vertical_requires_compatible_broker_and_matching_strategy_owner(tmp_path):
    from tests.trading.test_verticals import vertical

    class Spread(Strategy):
        def on_event(self, event, context):
            if isinstance(event, Timer):
                return vertical()

    async def run():
        with OrderJournal(tmp_path / "orders.sqlite") as journal:
            broker = FakeBroker(journal)
            async with bridge(broker, {"spread": Spread(), "impostor": Spread()}) as live:
                await live.step(Timer(now(), "entry"))
                assert not broker.sent
                reasons = [event.reason for event in live.events if isinstance(event, Rejection)]
                assert any("VerticalBroker" in reason for reason in reasons)
                assert any("identity" in reason for reason in reasons)
    asyncio.run(run())


def test_simulation_rejects_vertical_commands_without_option_accounting():
    from convexity.backtest import SimulatedExchange
    from convexity.runtime import Runtime
    from tests.trading.test_verticals import vertical

    class Spread(Strategy):
        def on_start(self, context):
            return vertical()

    instruments = {"A": Instrument("A")}
    live = Runtime({"spread": Spread()}, SimulatedExchange(instruments), instruments)
    live.start(now())
    assert any(isinstance(event, Rejection) and "unsupported" in event.reason for event in live.events)
    assert not live.orders and not live.fills and not live.positions


def test_runtime_can_cancel_its_owned_vertical_without_stock_conversion(tmp_path):
    from convexity.runtime import BrokerOrderUpdate, CancelOrder
    from tests.trading.test_verticals import vertical

    class Cancel(Strategy):
        def on_event(self, event, context):
            if isinstance(event, Timer):
                assert context.broker_orders[0].request == vertical()
                return CancelOrder("v1")

    async def run():
        with OrderJournal(tmp_path / "orders.sqlite") as journal:
            journal.intent(vertical(), "DU123", 1, 71, 0)
            journal.status("v1", "Submitted", D(0))
            broker = FakeBroker(journal)
            async with bridge(broker, {"spread": Cancel()}) as live:
                await live.step(Timer(now(), "cancel"))
                assert journal.get("v1").status == "Cancelled"
                assert any(isinstance(event, BrokerOrderUpdate) and event.record.status == "Cancelled"
                           for event in live.events)
                assert not broker.sent
    asyncio.run(run())


def test_refresh_reuses_fresh_quotes_but_submission_still_quotes_again(tmp_path):
    async def run():
        with OrderJournal(tmp_path / "orders.sqlite") as journal:
            broker = FakeBroker(journal)
            original = broker.quote
            calls = []
            async def counted(request):
                calls.append(request.symbol)
                return await original(request)
            broker.quote = counted
            async with bridge(broker, {"s": BuyOnTimer()}) as live:
                await live.refresh()
                await live.refresh()
                assert calls == ["A"]
                await live.step(Timer(now(), "entry"))
                assert calls == ["A", "A"]
                old = now() - timedelta(hours=1)
                live.quotes["A"] = replace(live.quotes["A"], bid_received_at=old, ask_received_at=old)
                await live.refresh()
                assert calls == ["A", "A", "A"]
    asyncio.run(run())


def test_batch_waits_for_broker_acknowledgment_before_routing_next_intent(tmp_path):
    class Pending(FakeBroker):
        def __init__(self, journal):
            super().__init__(journal)
            self.observed = []

        async def submit(self, request, *, submission_guard=None):
            if submission_guard is not None:
                submission_guard()
            assert all(not record.uncertain for record in self.journal.orders())
            self.sent.append(request)
            self.journal.intent(request, "DU123", len(self.sent), 71, 123)
            record = self.journal.get(request.client_order_id)
            assert self.on_update is not None
            self.on_update(record)
            return record

        async def observe(self, identity, *, seconds, until):
            assert until == "acknowledged"
            await asyncio.sleep(0)
            self.observed.append(identity)
            self.journal.status(identity, "Submitted", D(0))
            record = self.journal.get(identity)
            assert self.on_update is not None
            self.on_update(record)
            return record

    class Batch(Strategy):
        def on_event(self, event, context):
            if isinstance(event, Timer):
                return SubmitOrder("A", "BUY", 1), SubmitOrder("A", "BUY", 1)

    async def run():
        with OrderJournal(tmp_path / "orders.sqlite") as journal:
            broker = Pending(journal)
            async with bridge(broker, {"s": Batch()}) as live:
                await live.step(Timer(now(), "entry"))
                assert len(broker.sent) == len(broker.observed) == 2
                assert not live.errors
    asyncio.run(run())


@pytest.mark.parametrize("complete", [True, False])
def test_completed_combo_waits_for_real_leg_details_or_blocks_following_stock(tmp_path, complete):
    from convexity.brokers.ibkr_verticals import VerticalBroker
    from tests.trading.test_verticals import vertical

    class Combo(FakeBroker, VerticalBroker):
        def __init__(self, journal):
            FakeBroker.__init__(self, journal)
            self.strategy_budgets = {"spread": D(1000)}

        async def submit(self, request, *, submission_guard=None):
            if submission_guard is not None:
                submission_guard()
            if isinstance(request, StockOrder):
                assert self.journal.get("v1").executed_quantity == 1
                return await FakeBroker.submit(self, request)
            self.sent.append(request)
            self.journal.intent(request, "DU123", 1, 71, 0)
            self.journal.status(request.client_order_id, "Filled", D(1))
            record = self.journal.get(request.client_order_id)
            assert self.on_update is not None
            self.on_update(record)
            if complete:
                asyncio.create_task(self.complete_legs())
            return record

        async def complete_legs(self):
            await asyncio.sleep(.01)
            for contract, side in ((101, "BUY"), (102, "SELL")):
                self.journal.execution(Execution(str(contract), "v1", D(1), D(1), side,
                                                 now().isoformat(), contract_id=contract))
            assert self.on_update is not None
            self.on_update(self.journal.get("v1"))

    class Spread(Strategy):
        def on_event(self, event, context):
            if isinstance(event, Timer):
                return vertical()

    async def run():
        with OrderJournal(tmp_path / "orders.sqlite") as journal:
            broker = Combo(journal)
            async with bridge(broker, {"spread": Spread(), "stock": BuyOnTimer()},
                              acknowledgment_timeout=.2 if complete else .01) as live:
                if complete:
                    await live.step(Timer(now(), "entry"))
                else:
                    with pytest.raises(DeliveryError, match="unresolved"):
                        await live.step(Timer(now(), "entry"))
                assert len(broker.sent) == (2 if complete else 1)
                assert journal.get("v1").executed_quantity == (1 if complete else 0)
                if not complete:
                    assert journal.get("v1").uncertain
                    assert any("unresolved" in error.lower() for error in live.errors)
    asyncio.run(run())


def test_options_only_runtime_needs_balances_but_no_stock_quote_subscription(tmp_path):
    class NoStocks(FakeBroker):
        async def quote(self, request):
            pytest.fail("An options-only runtime must not request stock quotes.")

    async def run():
        with OrderJournal(tmp_path / "orders.sqlite") as journal:
            broker = NoStocks(journal)
            async with PaperRuntime(cast(PaperBroker, broker), {"options": Strategy()}, {}, budgets={"options": D(100)},
                                    max_order_notional=D(100), market_collar_bps=D(0)) as live:
                await live.refresh()
                assert live.context("options").cash == 10000
                assert live.context("options").marks == {}
                assert live.context("options").account_positions == {}
    asyncio.run(run())


def test_non_usd_base_account_preserves_ledger_and_converts_usd_risk(tmp_path):
    from convexity.runtime import AccountState

    class CanadianBroker(FakeBroker):
        async def snapshot(self):
            snapshot = await super().snapshot()
            return replace(snapshot, summary=(
                AccountValue("NetLiquidation", "28000", "CAD"),
                AccountValue("TotalCashValue", "14000", "CAD"),
                AccountValue("AvailableFunds", "1400", "CAD"),
                AccountValue("BuyingPower", "5600", "CAD"),
                AccountValue("InitMarginReq", "26600", "CAD"),
                AccountValue("MaintMarginReq", "20000", "CAD"),
                AccountValue("ExcessLiquidity", "8000", "CAD"),
                AccountValue("CashBalance", "1000", "USD"),
                AccountValue("CashBalance", "12600", "CAD"),
                AccountValue("ExchangeRate", "1.4", "USD"),
            ))

    async def exercise():
        with OrderJournal(tmp_path / "journal.sqlite3") as journal:
            broker = CanadianBroker(journal)
            async with bridge(broker, {"s": Strategy()}) as runtime:
                context = runtime.context("s")
                assert isinstance(context.account, AccountState)
                assert context.cash == 14000 and context.equity == 28000
                assert context.account.cash_balances == {"USD": 1000, "CAD": 12600}
                assert context.account.buying_power == 5600
                assert context.account.sizing_power == 1400
                assert context.account.equity_in("USD") == 20000
                assert context.price("A", side="BUY") == 100
                assert context.buy_quantities({"A": 1}) == {"A": 9}
                await runtime.command("s", SubmitOrder.market("A", "BUY", 4, client_order_id="entry"))
                assert len(broker.sent) == 1
                context = runtime.context("s")
                assert context.account is not None
                assert context.account.reserved > 0
                assert context.buy_quantities({"A": 1})["A"] < 10
    asyncio.run(exercise())
