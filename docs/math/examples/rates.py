"""curve-fitted Q rates and separate stochastic CPI laws."""

from convexity.math.curves import flat_curve
from convexity.math.rates import ShortRateModel, InflationDynamics

nominal = flat_curve("2026-01-01", 0.04)
model = ShortRateModel("hull_white", {"kappa": 0.1, "sigma": 0.01}, nominal)
print("5-year bond:", model.discount_bond(0.0, 5.0, 0.04))
physical_cpi = InflationDynamics(0.1, "P", drift=0.03)
pricing_cpi = InflationDynamics(
    0.1, "Q", nominal=nominal, real=flat_curve("2026-01-01", 0.01)
)
print("P expected CPI:", physical_cpi.transition(100.0, 0.0, 1.0).mean())
print("Q CPI call:", pricing_cpi.option(100.0, 100.0, 1.0))
