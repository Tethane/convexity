"""Read-only status keeps account/manual positions distinct from journal ownership."""
import asyncio
from datetime import datetime, timezone
from decimal import Decimal as D
from types import SimpleNamespace

from convexity.brokers import status
from convexity.brokers.ibkr import PaperAccountSnapshot, Position, _contract
from convexity.trading import Execution, OrderJournal, StockOrder
from tests.brokers.test_ibkr import contract, config

NOW = datetime(2026, 10, 9, 14, tzinfo=timezone.utc)


def record(journal, identity, owner, side, quantity, *, account="DU123", filled=True):
    request = StockOrder(identity, owner, "SPY", side, quantity, D(100))
    journal.intent(request, account, len(journal.orders()) + 1, 71, 123)
    journal.status(identity, "Filled", D(quantity))
    if filled:
        journal.execution(Execution(identity + "-execution", identity, D(quantity), D(100), side, NOW.isoformat()))


def test_paper_status_attributes_only_saved_executions_and_keeps_manual_difference(tmp_path, monkeypatch):
    snapshot = PaperAccountSnapshot("DU123", NOW, NOW, (), (Position(_contract(contract()), D(8), D(100)),), ())

    async def read(config):
        assert config.account == "DU123"
        return snapshot

    monkeypatch.setattr(status, "read_paper_account", read)
    with OrderJournal(tmp_path / "orders.sqlite") as journal:
        record(journal, "long", "alpha", "BUY", 5)
        record(journal, "short", "beta", "SELL", 2)
        record(journal, "other-account", "other", "BUY", 100, account="DU999")
        before = tuple(journal.events(order.request.client_order_id) for order in journal.orders())
        result = asyncio.run(status.read_paper_status(config(), journal))
        assert result.owned_positions == {"alpha": {123: D(5)}, "beta": {123: D(-2)}}
        assert result.unassigned_positions == {123: D(5)}
        assert not result.faults
        assert len(result.orders) == 2
        assert "DU123" not in result.format() and "DU999" not in result.format()
        assert "UNASSIGNED contract 123: +5" in result.format()
        assert tuple(journal.events(order.request.client_order_id) for order in journal.orders()) == before


def test_paper_status_surfaces_missing_execution_details_without_inventing_ownership(tmp_path, monkeypatch):
    snapshot = PaperAccountSnapshot("DU123", NOW, NOW, (), (Position(_contract(contract()), D(2), D(100)),), ())

    async def read(config):
        return snapshot

    monkeypatch.setattr(status, "read_paper_account", read)
    with OrderJournal(tmp_path / "orders.sqlite") as journal:
        record(journal, "missing", "alpha", "BUY", 2, filled=False)
        result = asyncio.run(status.read_paper_status(config(), journal))
        assert result.owned_positions == {"alpha": {123: D(0)}}
        assert result.unassigned_positions == {123: D(2)}
        assert any("missing execution details" in fault for fault in result.faults)
        assert not journal.get("missing").executions


def test_option_status_uses_leg_executions_and_exposes_missing_second_leg(tmp_path, monkeypatch):
    from dataclasses import replace
    from tests.trading.test_verticals import vertical

    base = _contract(contract())
    snapshot = PaperAccountSnapshot("DU123", NOW, NOW, (), (
        Position(replace(base, contract_id=101), D(1), D(100)),
        Position(replace(base, contract_id=102), D(-1), D(100))), ())

    async def read(config):
        return snapshot

    monkeypatch.setattr(status, "read_paper_account", read)
    with OrderJournal(tmp_path / "orders.sqlite") as journal:
        journal.intent(vertical(), "DU123", 1, 71, 0)
        journal.status("v1", "Filled", D(1))
        journal.execution(Execution("long", "v1", D(1), D(1), "BUY", NOW.isoformat(), contract_id=101))
        incomplete = asyncio.run(status.read_paper_status(config(), journal))
        assert incomplete.owned_positions == {"spread": {101: D(1), 102: D(0)}}
        assert incomplete.unassigned_positions == {102: D(-1)}
        assert any("missing execution details" in fault for fault in incomplete.faults)
        journal.execution(Execution("short", "v1", D(1), D('.5'), "SELL", NOW.isoformat(), contract_id=102))
        before = journal.events("v1")
        complete = asyncio.run(status.read_paper_status(config(), journal))
        assert complete.owned_positions == {"spread": {101: D(1), 102: D(-1)}}
        assert not complete.unassigned_positions and not complete.faults
        assert journal.events("v1") == before


def test_option_status_flags_aggregate_unbalanced_legs_until_reconciled(tmp_path, monkeypatch):
    from tests.trading.test_verticals import vertical

    async def read(config):
        return PaperAccountSnapshot("DU123", NOW, NOW, (), (), ())

    monkeypatch.setattr(status, "read_paper_account", read)
    with OrderJournal(tmp_path / "orders.sqlite") as journal:
        journal.intent(vertical(), "DU123", 1, 71, 0)
        journal.execution(Execution("orphan", "v1", D(1), D(1), "BUY", NOW.isoformat(), contract_id=101))
        journal.status("v1", "Cancelled", D(0))
        result = asyncio.run(status.read_paper_status(config(), journal))
        assert any("unbalanced vertical legs" in fault for fault in result.faults)
        journal.execution(Execution("late-wing", "v1", D(1), D('.5'), "SELL", NOW.isoformat(), contract_id=102))
        result = asyncio.run(status.read_paper_status(config(), journal))
        assert not any("unbalanced vertical legs" in fault for fault in result.faults)
        journal.intent(vertical("SELL", "v2"), "DU123", 2, 71, 0)
        journal.execution(Execution("close-long", "v2", D(1), D(1), "SELL", NOW.isoformat(), contract_id=101))
        journal.execution(Execution("close-wing", "v2", D(1), D('.5'), "BUY", NOW.isoformat(), contract_id=102))
        result = asyncio.run(status.read_paper_status(config(), journal))
        assert not result.faults
        assert result.owned_positions == {"spread": {101: D(0), 102: D(0)}}
