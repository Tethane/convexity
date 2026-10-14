from typing import Literal
from datetime import date
from decimal import Decimal as D

import pytest

from convexity.trading import Execution, OptionIdentity, OrderJournal, VerticalOrder


def vertical(side: Literal["BUY", "SELL"] = "BUY", local_id="v1"):
    return VerticalOrder(
        local_id,
        "spread",
        OptionIdentity(101, "SPY", date(2026, 10, 9), D(777), "C", "SPY"),
        OptionIdentity(102, "SPY", date(2026, 10, 9), D(778), "C", "SPY"),
        side,
        1,
        D(".50"),
    )


def test_legs_are_one_combo_not_two_and_reopen_preserves_identity(tmp_path):
    path = tmp_path / "j.sqlite"
    with OrderJournal(path) as journal:
        request = vertical()
        journal.intent(request, "DU123", 1, 71, 0)
        a = Execution(
            "a",
            "v1",
            D(1),
            D("1.25"),
            "BUY",
            "2026-10-09T14:00:00+00:00",
            contract_id=101,
        )
        b = Execution(
            "b",
            "v1",
            D(1),
            D(".75"),
            "SELL",
            "2026-10-09T14:00:00+00:00",
            contract_id=102,
        )
        journal.execution(a)
        assert journal.get("v1").executed_quantity == 0
        journal.execution(b)
        assert journal.get("v1").executed_quantity == 1
        assert journal.get("v1").filled_quantity == 1
        journal.status(
            "v1", "Filled", D(0)
        )  # completed-order replay can omit cumulative quantity
        assert journal.get("v1").filled_quantity == 1
        assert not journal.execution(a)
        assert journal.attributed_position("DU123", "spread", 101) == 1
        assert journal.attributed_position("DU123", "spread", 102) == -1
        journal.intent(vertical("SELL", "v2"), "DU123", 2, 71, 0)
        journal.execution(
            Execution(
                "c",
                "v2",
                D(1),
                D("1.30"),
                "SELL",
                "2026-10-09T14:01:00+00:00",
                contract_id=101,
            )
        )
        journal.execution(
            Execution(
                "d",
                "v2",
                D(1),
                D(".80"),
                "BUY",
                "2026-10-09T14:01:00+00:00",
                contract_id=102,
            )
        )
        assert journal.attributed_position("DU123", "spread", 101) == 0
        assert journal.attributed_position("DU123", "spread", 102) == 0
    with OrderJournal(path) as journal:
        assert journal.get("v1").request == request
        assert len(journal.get("v1").executions) == 2


def test_reject_unowned_leg_wrong_side_and_leg_overfill(tmp_path):
    with OrderJournal(tmp_path / "j.sqlite") as journal:
        journal.intent(vertical(), "DU123", 1, 71, 0)
        for cid, side, quantity in ((999, "BUY", 1), (102, "BUY", 1), (101, "BUY", 2)):
            with pytest.raises(ValueError):
                journal.execution(
                    Execution(
                        "x",
                        "v1",
                        D(quantity),
                        D(1),
                        side,
                        "2026-10-09T14:00:00+00:00",
                        contract_id=cid,
                    )
                )
        assert journal.get("v1").executions == ()
