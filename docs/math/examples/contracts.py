"""scheduled cash flows and equivalent settlement value."""

from convexity.math.contracts import (
    Exercise,
    VanillaContract,
    schedule,
    fixed_cashflows,
)

for flow in fixed_cashflows(schedule("2026-01-01", "2028-01-01"), 0.05):
    print(flow.date.ISO(), flow.amount)
contract = VanillaContract(
    100.0,
    Exercise("european", ("2027-01-01",)),
    multiplier=100.0,
    settlement="physical",
)
print("Exercise delivery (shares, cash):", contract.delivery(110.0))
