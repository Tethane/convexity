"""Family 26: reduced-form Q value and a distinct structural default model."""

from convexity.math.curves import flat_curve
from convexity.math.credit import (
    flat_hazard,
    risky_cashflows,
    merton_structural,
    estimate_hazard,
)

reference = "2026-01-01"
curve, hazard = flat_curve(reference, 0.04), flat_hazard(reference, 0.02)
print(
    "Risky zero:",
    risky_cashflows(
        [3.0], [100.0], curve, hazard, recovery=0.4, recovery_notional=100.0
    ),
)
print("P hazard estimate:", estimate_hazard([2.0, 3.0, 5.0], [True, False, True]))
print(
    "Structural:", merton_structural(120.0, 100.0, 0.2, 1.0, 0.04, physical_drift=0.1)
)
