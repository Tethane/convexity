"""Recurring SPY overnight holds with typed schedules and one order ID per decision."""

from datetime import date, time, timedelta
from decimal import Decimal

import convexity as cx


class OvernightHold(cx.Strategy):
    metadata = cx.StrategyMetadata("overnight_hold", version="2")
    contract = cx.StrategyContract(
        instruments=("SPY",),
        subscriptions=(
            cx.EventSubscription(
                cx.Timer, schedule=cx.ClockSchedule("enter", at=time(15, 50))
            ),
            cx.EventSubscription(
                cx.Timer, schedule=cx.ClockSchedule("exit", at=time(9, 30))
            ),
            cx.EventSubscription(cx.Rejection),
        ),
    )

    def on_timer(self, event, context):
        # Stable for retries of this decision, unique for the next session.
        order_id = f"{event.name}:{event.time.isoformat()}"
        if (
            event.name == "enter"
            and not context.position("SPY")
            and not context.pending("SPY")
        ):
            price = context.price("SPY", max_age=timedelta(days=4))
            account = context.account
            if (
                account is None
                or account.available_funds is None
                or not account.cash_balances_complete
            ):
                raise ValueError(
                    "Complete cash balances and available funds are required."
                )
            budget = max(
                Decimal(0),
                min(
                    account.cash_balances.get("USD", Decimal(0)),
                    account.available_funds,
                )
                - account.reserved
                - Decimal("10"),
            )
            shares = int(budget // (price * Decimal("1.01")))
            if shares:
                return cx.SubmitOrder.market(
                    "SPY", "BUY", shares, time_in_force="DAY", client_order_id=order_id
                )
        if (
            event.name == "exit"
            and context.position("SPY") > 0
            and not context.pending("SPY")
        ):
            return cx.SubmitOrder.market(
                "SPY",
                "SELL",
                context.position("SPY"),
                time_in_force="DAY",
                client_order_id=order_id,
            )

    def on_rejection(self, event, context):
        # This example stops visibly instead of hiding a rejected daily entry.
        raise RuntimeError(f"Order rejected at {event.time}: {event.reason}")


if __name__ == "__main__":
    result = cx.Research(date(2025, 1, 2), date(2026, 10, 12)).run(
        OvernightHold,
        initial_cash=100_000,
        base_currency="USD",
    )
    print(result.metrics, result.positions, result.open_orders)
