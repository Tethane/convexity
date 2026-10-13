"""Family 18: deterministic carry and deposit bootstrap, using only Convexity."""

from convexity.math.curves import bootstrap, deposit_quote, flat_curve, forward_price

reference = "2026-01-01"
curve = bootstrap(reference, [deposit_quote(0.04, "1Y")])
print("Bootstrapped discount:", curve.discount(1.0))
print("Quote residuals:", curve.calibration_errors)
print(
    "Forward:", forward_price(100.0, 1.0, curve, dividend=flat_curve(reference, 0.01))
)
