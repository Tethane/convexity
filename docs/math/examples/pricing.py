"""Family 19: one contract, three QuantLib engines."""

from convexity.math.contracts import VanillaContract, Exercise
from convexity.math.curves import flat_curve
from convexity.math.pricing import (
    BlackScholesModel,
    PricingProblem,
    VanillaEngine,
    price,
)

reference = "2026-01-01"
model = BlackScholesModel(
    100.0, flat_curve(reference, 0.05), flat_curve(reference, 0.02), 0.2
)
contract = VanillaContract(100.0, Exercise("european", ("2027-01-01",)))
problem = PricingProblem(contract, model, reference)
for method in ("analytic", "tree", "fd"):
    result = price(problem, VanillaEngine(method))
    print(method, result.value, dict(result.greeks))
