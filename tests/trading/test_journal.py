from typing import Literal
from decimal import Decimal

import pytest

from convexity.trading import Execution, OrderJournal, StockOrder


def order(local_id="order-1", side: Literal["BUY", "SELL"]="BUY", quantity=2):
    return StockOrder(local_id, "smoke", "SPY", side, quantity, Decimal("700.01"), "overnight")


def fill(exec_id="exec-1", qty="1", local_id="order-1", side: Literal["BUY", "SELL"]="BUY"):
    return Execution(exec_id, local_id, Decimal(qty), Decimal("700"), side, "2026-10-09T05:00:00+00:00")


def test_intent_durable_idempotent_and_conflicts(tmp_path):
    path = tmp_path / "orders.sqlite"
    with OrderJournal(path) as journal:
        assert journal.intent(order(), "DU123", 7, 71, 756733)
        assert not journal.intent(order(), "DU123", 99, 71, 756733)
        with pytest.raises(ValueError):
            journal.intent(order(side="SELL"), "DU123", 99, 71, 756733)
    with OrderJournal(path) as journal:
        row = journal.get("order-1")
        assert row.uncertain and not row.acknowledged
        assert row.broker_order_id == 7
        assert row.request.limit_price == Decimal("700.01")
        assert journal.events("order-1")[0]["kind"] == "intent"


def test_partial_duplicate_fill_cancel_race_and_fee(tmp_path):
    with OrderJournal(tmp_path / "orders.sqlite") as j:
        j.intent(order(), "DU123", 7, 71, 756733)
        j.status("order-1", "Submitted", Decimal(0), 100)
        j.cancel_requested("order-1")
        assert j.execution(fill())
        assert not j.execution(fill())
        row = j.get("order-1")
        assert row.status == "PartiallyFilled"
        assert row.cancel_pending
        assert row.filled_quantity == row.remaining_quantity == 1
        j.status("order-1", "Cancelled", Decimal(1), 100)
        assert j.get("order-1").status == "Cancelled"
        assert j.execution(fill("exec-2"))  # Late fill after cancel acknowledgment.
        j.status("order-1", "Submitted", Decimal(1), 100)  # Late stale status.
        assert j.get("order-1").status == "Filled"
        assert not j.get("order-1").cancel_pending
        j.commission("DU123", "exec-1", Decimal("0.35"), "USD")
        assert j.get("order-1").executions[0].commission == Decimal("0.35")
        assert j.attributed_position("DU123", "smoke", 756733) == 2
        assert j.attributed_position("DU123", "other", 756733) == 0


def test_status_fill_and_execution_do_not_double_count(tmp_path):
    with OrderJournal(tmp_path / "orders.sqlite") as j:
        j.intent(order(), "DU123", 7, 71, 756733)
        j.status("order-1", "Submitted", Decimal(1), 100)
        j.execution(fill())
        assert j.get("order-1").filled_quantity == 1
        assert j.attributed_position("DU123", "smoke", 756733) == 1
        with pytest.raises(ValueError, match="Conflicting"):
            j.execution(fill(qty="2"))
        assert len(j.get("order-1").executions) == 1


def test_rejection_after_library_cancel_echo_is_visible(tmp_path):
    with OrderJournal(tmp_path / "orders.sqlite") as j:
        j.intent(order(), "DU123", 7, 71, 756733)
        j.status("order-1", "Cancelled", Decimal(0))
        j.status("order-1", "Rejected", Decimal(0))
        assert j.get("order-1").status == "Rejected"


def test_pending_cancel_is_not_cancellation_or_ack(tmp_path):
    with OrderJournal(tmp_path / "orders.sqlite") as j:
        j.intent(order(), "DU123", 7, 71, 756733)
        j.cancel_requested("order-1")
        j.status("order-1", "PendingCancel", Decimal(0))
        row = j.get("order-1")
        assert not row.terminal and not row.acknowledged and row.uncertain
        j.cancel_rejected("order-1")
        assert not j.get("order-1").cancel_pending
        assert not j.get("order-1").terminal


@pytest.mark.parametrize("kwargs", [dict(quantity=0), dict(quantity=1.5), dict(side="BOUGHT"),
    dict(limit_price=Decimal("NaN")), dict(limit_price=1.2), dict(session="all"),
    dict(client_order_id=""), dict(currency="EUR")])
def test_invalid_order(kwargs):
    data = dict(client_order_id="one", strategy_id="smoke", symbol="SPY", side="BUY", quantity=1, limit_price=Decimal("700"))
    with pytest.raises(ValueError):
        StockOrder(**(data | kwargs))


def test_fill_deduplication_survives_reopen(tmp_path):
    path = tmp_path / 'orders.sqlite'
    with OrderJournal(path) as journal:
        journal.intent(order(), 'DU123', 7, 71, 756733)
        journal.execution(fill())
    with OrderJournal(path) as journal:
        assert not journal.execution(fill())
        assert journal.attributed_position('DU123', 'smoke', 756733) == 1
        assert len(journal.get('order-1').executions) == 1


def test_execution_validation_preserves_attribution_and_quantity(tmp_path):
    from dataclasses import replace
    with OrderJournal(tmp_path / "orders.sqlite") as j:
        j.intent(order(), "DU123", 7, 71, 756733)
        for invalid in (fill(side="SELL"), fill(qty="3"),
                        replace(fill(), price=Decimal("-1")),
                        replace(fill(), quantity=Decimal("NaN")),
                        replace(fill(), occurred_at="2026-10-09T05:00:00")):
            with pytest.raises(ValueError):
                j.execution(invalid)
        assert not j.get("order-1").executions
        assert j.get("order-1").uncertain
        assert j.execution(fill())
        with pytest.raises(ValueError, match="exceed"):
            j.execution(fill("exec-2", qty="2"))
        assert j.attributed_position("DU123", "smoke", 756733) == 1


def test_supplied_execution_commission_survives_reopen(tmp_path):
    from dataclasses import replace
    path = tmp_path / "orders.sqlite"
    with OrderJournal(path) as j:
        j.intent(order(), "DU123", 7, 71, 756733)
        j.execution(replace(fill(), commission=Decimal("0.35"), commission_currency="USD"))
    with OrderJournal(path) as j:
        execution = j.get("order-1").executions[0]
        assert execution.commission == Decimal("0.35")
        assert execution.commission_currency == "USD"


def test_invalid_reported_fill_does_not_make_order_terminal(tmp_path):
    with OrderJournal(tmp_path / "orders.sqlite") as j:
        j.intent(order(), "DU123", 7, 71, 756733)
        for status, quantity in (("Submitted", Decimal("3")), ("Filled", Decimal("0")),
                                 ("Submitted", Decimal("NaN"))):
            with pytest.raises(ValueError):
                j.status("order-1", status, quantity)
        assert not j.get("order-1").terminal
        assert j.get("order-1").filled_quantity == 0


def test_intent_creation_time_survives_reopen_for_session_trade_limits(tmp_path):
    path = tmp_path / "orders.sqlite"
    with OrderJournal(path) as journal:
        journal.intent(order(), "DU123", 7, 71, 756733)
        created = journal.get("order-1").created_at
        assert created is not None and created.utcoffset() is not None
    with OrderJournal(path) as journal:
        assert journal.get("order-1").created_at == created


def test_legacy_stock_request_keeps_day_default_and_original_evidence(tmp_path):
    import json
    import sqlite3
    path = tmp_path / "legacy.sqlite"
    with OrderJournal(path) as journal:
        journal.intent(order(), "DU123", 7, 71, 756733)
        evidence = journal.events("order-1")
    with sqlite3.connect(path) as db:
        payload = json.loads(db.execute("SELECT request FROM orders").fetchone()[0])
        payload.pop("time_in_force")
        old_payload = json.dumps(payload, sort_keys=True)
        db.execute("UPDATE orders SET request=?", (old_payload,))
        db.execute("DROP TABLE runtime_intents")
    with OrderJournal(path) as journal:
        request = journal.get("order-1").request
        assert isinstance(request, StockOrder)
        assert request.time_in_force == "DAY"
        assert not journal.intent(order(), "DU123", 7, 71, 756733)
        assert journal.events("order-1") == evidence
        assert journal.runtime_intent("order-1") is None
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT request FROM orders").fetchone()[0] == old_payload
