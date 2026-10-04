"""Family 27: weighted tail risk, Euler allocation and whole-path breaches."""

import numpy as np
from convexity.math import risk, simulation

losses = np.array([-3.0, 1.0, 7.0, 9.0])
weights = [0.3, 0.4, 0.2, 0.1]
print(
    "Weighted VaR/ES:",
    risk.value_at_risk(losses, 0.5, weights=weights),
    risk.expected_shortfall(losses, 0.5, weights=weights),
)
contributions = risk.es_contributions([2.0, -1.0], [[1, 0], [0, 2], [3, 1]], 0.5)
print("ES components:", contributions.component, "total:", contributions.value)
paths = simulation.PathScenarios(
    [0.0, 1.0, 2.0], [[100.0, 110.0, 95.0], [100.0, 90.0, 120.0]], [0.7, 0.3]
)
print("Maximum drawdowns:", risk.maximum_drawdown(paths.values))
print("Grid breach probability:", risk.breach_probability(paths, 92.0))
print("Stress margin:", risk.scenario_margin(risk.stress(losses, scale=1.5), addon=2.0))
