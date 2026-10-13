import asyncio
from datetime import datetime, timezone
from decimal import Decimal as D
from types import SimpleNamespace as NS

import pytest

from convexity.brokers import ibkr_trading
from convexity.brokers.ibkr import PaperAccountError
from convexity.brokers.ibkr_verticals import VerticalBroker
from convexity.trading import OrderJournal
from tests.brokers.test_ibkr import config, pytestmark
from tests.brokers.test_ibkr_trading import TradingClient
from tests.trading.test_verticals import vertical


class ComboClient(TradingClient):
    def __init__(self, journal):
        super().__init__(journal)
        self.positions = []
        self.summary = [NS(account="DU123", tag="AvailableFunds", value="1000", currency="USD")]

    def placeOrder(self, contract, order):
        assert self.journal.get(order.orderRef).uncertain
        assert contract.secType == "BAG"
        assert [(l.conId, l.ratio, l.action) for l in contract.comboLegs] == [(101, 1, "BUY"), (102, 1, "SELL")]
        self.sent.append(order)
        order.clientId = 71
        return NS(contract=contract, order=order, orderStatus=NS(status="PendingSubmit", filled=0, permId=0))


class TestBroker(VerticalBroker):
    __test__ = False
    async def _qualified_legs(self, request):
        return NS(conId=101), NS(conId=102)

    async def vertical_quotes(self, request):
        now = datetime.now(timezone.utc)
        return (NS(status="received", bid_received_at=now, ask_received_at=now, max_age=30),
                NS(status="received", bid_received_at=now, ask_received_at=now, max_age=30))


def test_combo_intent_before_send_leg_fill_accounting_and_no_double_count(tmp_path, monkeypatch):
    with OrderJournal(tmp_path / "j.sqlite") as journal:
        client = ComboClient(journal)
        monkeypatch.setattr(ibkr_trading, "_new_client", lambda: client)
        async def run():
            async with TestBroker(config(), journal, strategy_budgets={"spread": D(100)}) as broker:
                await broker.submit(vertical())
                assert len(client.sent) == 1
                await broker.submit(vertical())
                assert len(client.sent) == 1
                for sec, cid, side, exec_id in (("BAG", 0, "BOT", "summary"), ("OPT", 101, "BOT", "long"), ("OPT", 102, "SLD", "short")):
                    broker._on_fill(None, NS(contract=NS(secType=sec, conId=cid), execution=NS(
                        acctNumber="DU123", clientId=71, orderId=7, permId=0, orderRef="v1", side=side,
                        execId=exec_id, shares=1, price=1, time=datetime.now(timezone.utc))))
                broker._check()
                row = journal.get("v1")
                assert len(row.executions) == 2
                assert row.executed_quantity == row.filled_quantity == 1
        asyncio.run(run())


def test_budget_and_unowned_close_never_reach_transport(tmp_path, monkeypatch):
    with OrderJournal(tmp_path / "j.sqlite") as journal:
        client = ComboClient(journal)
        monkeypatch.setattr(ibkr_trading, "_new_client", lambda: client)
        async def run():
            async with TestBroker(config(), journal, strategy_budgets={"spread": D(50)}) as broker:
                with pytest.raises(PaperAccountError, match="budget"):
                    await broker.submit(vertical())
                with pytest.raises(PaperAccountError, match="owned"):
                    await broker.submit(vertical("SELL"))
                assert not client.sent
                assert not journal.orders()
        asyncio.run(run())


class FlexibleComboClient(ComboClient):
    def __init__(self, journal):
        from itertools import count
        super().__init__(journal)
        ids = count(7)
        self.client.getReqId = lambda: next(ids)

    def placeOrder(self, contract, order):
        record = self.journal.get(order.orderRef)
        assert record.uncertain
        assert [leg.conId for leg in contract.comboLegs] == [record.request.long_leg.contract_id, record.request.short_leg.contract_id]
        self.sent.append(order)
        order.clientId = 71
        return NS(contract=contract, order=order, orderStatus=NS(status="PendingSubmit", filled=0, permId=0))


class FlexibleBroker(TestBroker):
    __test__ = False
    async def _qualified_legs(self, request):
        return NS(conId=request.long_leg.contract_id), NS(conId=request.short_leg.contract_id)


def second_vertical():
    from dataclasses import replace
    request = vertical(local_id="qqq")
    return replace(request, long_leg=replace(request.long_leg, contract_id=201, symbol="QQQ", trading_class="QQQ"),
                   short_leg=replace(request.short_leg, contract_id=202, symbol="QQQ", trading_class="QQQ"))


@pytest.mark.parametrize("budget,accepted", [(200, True), (150, False)])
def test_disjoint_pairs_share_owner_but_reserve_pending_width(tmp_path, monkeypatch, budget, accepted):
    with OrderJournal(tmp_path / "j.sqlite") as journal:
        client = FlexibleComboClient(journal)
        monkeypatch.setattr(ibkr_trading, "_new_client", lambda: client)
        async def run():
            async with FlexibleBroker(config(), journal, strategy_budgets={"spread": D(budget)}) as broker:
                await broker.submit(vertical())
                journal.status("v1", "Submitted", D(0))  # Acknowledged, still unfilled.
                if accepted:
                    await broker.submit(second_vertical())
                    assert len(client.sent) == 2
                    assert {record.broker_order_id for record in journal.orders()} == {7, 8}
                else:
                    with pytest.raises(PaperAccountError, match="width exposure"):
                        await broker.submit(second_vertical())
                    assert len(client.sent) == 1
                    assert len(journal.orders()) == 1
        asyncio.run(run())


def test_overlapping_pending_leg_remains_blocked(tmp_path, monkeypatch):
    from dataclasses import replace
    first = vertical()
    overlapping = replace(first, client_order_id="overlap", long_leg=first.short_leg,
                          short_leg=replace(first.short_leg, contract_id=103, strike=first.short_leg.strike + 1))
    with OrderJournal(tmp_path / "j.sqlite") as journal:
        client = FlexibleComboClient(journal)
        monkeypatch.setattr(ibkr_trading, "_new_client", lambda: client)
        async def run():
            async with FlexibleBroker(config(), journal, strategy_budgets={"spread": D(1000)}) as broker:
                await broker.submit(first)
                journal.status("v1", "Submitted", D(0))
                with pytest.raises(PaperAccountError, match="working orders on these option legs"):
                    await broker.submit(overlapping)
                assert len(client.sent) == 1
        asyncio.run(run())


def test_separate_strategy_budgets_do_not_double_spend_account_funds(tmp_path, monkeypatch):
    from dataclasses import replace
    with OrderJournal(tmp_path / "j.sqlite") as journal:
        client = FlexibleComboClient(journal)
        client.summary[0].value = "75"
        monkeypatch.setattr(ibkr_trading, "_new_client", lambda: client)
        async def run():
            async with FlexibleBroker(config(), journal, strategy_budgets={"spread": D(100), "other": D(100)}) as broker:
                await broker.submit(vertical())
                journal.status("v1", "Submitted", D(0))
                with pytest.raises(PaperAccountError, match="available funds"):
                    await broker.submit(replace(second_vertical(), strategy_id="other"))
                assert len(client.sent) == 1
        asyncio.run(run())


def test_explicit_close_needs_balanced_owned_legs_but_no_quotes(tmp_path, monkeypatch):
    from convexity.trading import Execution
    from dataclasses import replace
    with OrderJournal(tmp_path / "j.sqlite") as journal:
        client = ComboClient(journal)
        monkeypatch.setattr(ibkr_trading, "_new_client", lambda: client)
        entry = vertical()
        journal.intent(entry, "DU123", 5, 71, 0)
        for cid, side in ((101, "BUY"), (102, "SELL")):
            journal.execution(Execution(str(cid), "v1", D(1), D(1), side,
                datetime.now(timezone.utc).isoformat(), contract_id=cid))
        async def run():
            async with TestBroker(config(), journal, strategy_budgets={"spread": D(100)}) as broker:
                async def unavailable(request):
                    raise RuntimeError("market data unavailable")
                async def positions():
                    return (NS(contract=NS(contract_id=101), quantity=D(1)),
                            NS(contract=NS(contract_id=102), quantity=D(-1)))
                broker.vertical_quotes = unavailable
                monkeypatch.setattr(broker, "positions", positions)
                with pytest.raises(ValueError, match="SELL"):
                    await broker.close_at_limit(entry)
                close = replace(entry, client_order_id="exit", side="SELL", limit_price=D(".01"))
                with pytest.raises(RuntimeError, match="market data"):
                    await broker.submit(close)
                await broker.close_at_limit(close)
                assert len(client.sent) == 1
                assert "market quotes not required" in journal.get("exit").diagnostics[-1]
        asyncio.run(run())


def test_vertical_submission_guard_runs_after_all_awaited_preflight(tmp_path, monkeypatch):
    with OrderJournal(tmp_path / "j.sqlite") as journal:
        client = ComboClient(journal)
        monkeypatch.setattr(ibkr_trading, "_new_client", lambda: client)
        preflight = []
        class DelayedBroker(TestBroker):
            async def _qualified_legs(self, request):
                await asyncio.sleep(0)
                preflight.append("qualified")
                return await super()._qualified_legs(request)

            async def positions(self):
                result = await super().positions()
                preflight.append("positions")
                return result

        def expired():
            assert preflight[-2:] == ["qualified", "positions"]
            raise PaperAccountError("Decision history expired after preflight")

        async def run():
            async with DelayedBroker(config(), journal, strategy_budgets={"spread": D(100)}) as broker:
                with pytest.raises(PaperAccountError, match="expired"):
                    await broker.submit(vertical(), submission_guard=expired)
                assert not client.sent and not journal.orders()
        asyncio.run(run())


@pytest.mark.parametrize("rate,accepted", [("1.4", True), (None, False)])
def test_foreign_base_funds_require_observed_usd_conversion(tmp_path, monkeypatch, rate, accepted):
    with OrderJournal(tmp_path / "j.sqlite") as journal:
        client = ComboClient(journal)
        client.summary = [NS(account="DU123", tag="AvailableFunds", value="1400", currency="CAD")]
        if rate is not None:
            client.summary.append(NS(account="DU123", tag="ExchangeRate", value=rate, currency="USD"))
        monkeypatch.setattr(ibkr_trading, "_new_client", lambda: client)
        async def run():
            async with TestBroker(config(), journal, strategy_budgets={"spread": D(100)}) as broker:
                if accepted:
                    await broker.submit(vertical())
                    assert len(client.sent) == 1
                else:
                    with pytest.raises(PaperAccountError, match="exchange rate"):
                        await broker.submit(vertical())
                    assert not client.sent
        asyncio.run(run())
