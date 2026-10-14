"""Smoke acceptance requires both 0-DTE underlyings and broker-verified flatness."""

from datetime import date, datetime, timezone
from decimal import Decimal as D
from types import SimpleNamespace
from typing import Any, Literal, cast

from convexity.trading import (
    Execution,
    OptionIdentity,
    OrderJournal,
    StockOrder,
    VerticalOrder,
)
from examples.paper_mvp import acceptance_faults


def populate(journal, *, qqq=True) -> dict[str, Any]:
    owners = ["monthly", "spread", "dte"]
    for symbol, contract, side in (("IWM", 1, "BUY"), ("QQQ", 2, "SELL")):
        for index, action in enumerate((side, "SELL" if side == "BUY" else "BUY")):
            identity = symbol + str(index)
            request = StockOrder(
                identity,
                "monthly",
                symbol,
                cast(Literal["BUY", "SELL"], action),
                1,
                D(100),
            )
            journal.intent(request, "DU123", len(journal.orders()) + 1, 71, contract)
            journal.execution(
                Execution(
                    identity,
                    identity,
                    D(1),
                    D(100),
                    action,
                    datetime.now(timezone.utc).isoformat(),
                )
            )
    legs = {}
    for key, owner, symbol, cid, expiry in (
        ("spread", "spread", "SPY", 10, date(2026, 10, 16)),
        ("dte_spy", "dte", "SPY", 20, date(2026, 10, 9)),
        ("dte_qqq", "dte", "QQQ", 30, date(2026, 10, 9)),
    ):
        pair = (
            OptionIdentity(cid, symbol, expiry, D(100), "C", symbol),
            OptionIdentity(cid + 1, symbol, expiry, D(101), "C", symbol),
        )
        legs[key] = [
            dict(symbol=leg.symbol, contract_id=leg.contract_id) for leg in pair
        ]
        if key == "dte_qqq" and not qqq:
            continue
        for side in ("BUY", "SELL"):
            identity = key + side
            request = VerticalOrder(identity, owner, *pair, side, 1, D(".50"))
            journal.intent(request, "DU123", len(journal.orders()) + 1, 71, 0)
            for leg in pair:
                journal.execution(
                    Execution(
                        identity + str(leg.contract_id),
                        identity,
                        D(1),
                        D(1),
                        request.leg_side(leg.contract_id),
                        datetime.now(timezone.utc).isoformat(),
                        contract_id=leg.contract_id,
                    )
                )
    return dict(owners=owners, stock_rule=dict(universe=["IWM", "QQQ"]), legs=legs)


def test_one_underlying_does_not_complete_grouped_zero_dte_strategy(tmp_path):
    with OrderJournal(tmp_path / "j.sqlite") as journal:
        plan = populate(journal, qqq=False)
        faults = acceptance_faults(plan, journal.orders(), journal, "DU123", (), ())
        assert any("dte QQQ requires actual BUY and SELL" in fault for fault in faults)


def test_all_underlyings_and_directions_with_flat_broker_positions_complete(tmp_path):
    with OrderJournal(tmp_path / "j.sqlite") as journal:
        plan = populate(journal)
        assert not acceptance_faults(plan, journal.orders(), journal, "DU123", (), ())
        assert any(
            "restart" in fault
            for fault in acceptance_faults(
                plan, journal.orders(), journal, "DU123", (), (), require_restart=True
            )
        )


def test_broker_position_difference_prevents_false_flat_completion(tmp_path):
    with OrderJournal(tmp_path / "j.sqlite") as journal:
        plan = populate(journal)
        actual = (
            SimpleNamespace(contract=SimpleNamespace(contract_id=20), quantity=D(1)),
        )
        faults = acceptance_faults(plan, journal.orders(), journal, "DU123", actual, ())
        assert any("Broker position for contract 20" in fault for fault in faults)


def test_extra_universe_members_do_not_require_unranked_stock_trades(tmp_path):
    from examples.paper_mvp import entry_faults

    with OrderJournal(tmp_path / "j.sqlite") as journal:
        plan = populate(journal)
        plan["stock_rule"]["universe"].append("UNRANKED")
        assert not entry_faults(plan, journal.orders())
        assert not acceptance_faults(plan, journal.orders(), journal, "DU123", (), ())


def test_close_only_still_requires_flat_reconciled_ownership(tmp_path):
    with OrderJournal(tmp_path / "j.sqlite") as journal:
        plan = populate(journal, qqq=False)
        assert not acceptance_faults(
            plan, journal.orders(), journal, "DU123", (), (), require_trades=False
        )
        actual = (
            SimpleNamespace(contract=SimpleNamespace(contract_id=20), quantity=D(1)),
        )
        assert acceptance_faults(
            plan, journal.orders(), journal, "DU123", actual, (), require_trades=False
        )
