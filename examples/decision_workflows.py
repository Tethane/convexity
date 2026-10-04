"""Physical scenarios -> Q premium -> financed wealth -> risk/utility decision.

One-year European call held to expiry, one-period finite action enumeration.
The physical law is supplied independently of the pricing drift. No trading IO.
"""

import numpy as np
from convexity.math import contracts, measures, pricing, simulation
from convexity.math import risk, utility, optimization, execution, wealth, policies

rng = np.random.default_rng(270036)
pair = measures.gbm_pair(mu=0.08, rate=0.04, dividend=0.0, sigma=0.2)
model = pricing.BlackScholesModel.from_measure_pair(pair, 100.0, "2026-01-01")
claim = contracts.VanillaContract(
    100.0, contracts.Exercise("european", ("2027-01-01",))
)
premium = pricing.price(
    pricing.PricingProblem(claim, model, "2026-01-01"), pricing.VanillaEngine()
).value
# These are P outcomes, never scenario-pricing inputs labeled Q implicitly.
terminal_spot = 100 * np.exp((0.08 - 0.5 * 0.2**2) + 0.2 * rng.normal(size=5000))
paths = simulation.PathScenarios(
    [0.0, 1.0], np.column_stack([np.full(5000, 100.0), terminal_spot]), sampling="iid"
)
payoffs = np.maximum(paths.values[:, -1] - 100.0, 0.0)
cost = execution.QuadraticCost([0.05], [[0.001]])
initial = wealth.Account([0.0], 100.0)
utility_model = utility.SmoothUtility(0.8, 50.0)


def outcome(quantity):
    after = wealth.trade(initial, [quantity], [premium], cost=cost([quantity]))
    funded = wealth.accrue(
        after, lending_factor=np.exp(0.04), borrowing_factor=np.exp(0.06)
    )
    terminal_wealth = funded.cash + quantity * payoffs
    gain = wealth.financed_gain(100.0, terminal_wealth, funding_factor=np.exp(0.04))
    return terminal_wealth, gain


def objective(action):
    return -utility.expected_utility(
        outcome(action[0])[0], utility_model, weights=paths.weights
    )


def feasible(action):
    return bool(
        risk.expected_shortfall(-outcome(action[0])[1], 0.95, weights=paths.weights)
        <= 20.0
    )


result = optimization.enumerate_minimum(objective, [range(6)], feasible=feasible)
values = np.array([-objective([q]) for q in range(6)])
allowed = np.array([feasible([q]) for q in range(6)])
decision = policies.no_action(values, feasible=allowed, switching=0.1)
print("Q premium:", premium)
print("Risk-feasible utility optimum:", result.solution)
print("Decision with inertia:", decision)
print("P tail risk:", risk.expected_shortfall(-outcome(decision.action)[1], 0.95))
