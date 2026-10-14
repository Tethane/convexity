"""yield analytics and key-rate exposure per 100 outstanding."""

from convexity.math.contracts import schedule
from convexity.math.curves import flat_curve
from convexity.math.fixed_income import fixed_bond, bond_analytics, key_rate_duration

curve = flat_curve("2026-01-01", 0.04)
bond = fixed_bond(schedule("2025-01-01", "2031-01-01"), 0.05)
print(bond_analytics(bond, curve, settlement_date="2026-03-01"))
print(
    "Key-rate durations:",
    key_rate_duration(bond, curve, ["2026-01-01", "2028-01-01", "2031-01-01"]),
)
