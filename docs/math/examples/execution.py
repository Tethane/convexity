"""Family 33: liquidation schedules and generated binomial assignment data."""

import numpy as np
from convexity.math import execution

for aversion in (0.0, 2.0, 10.0):
    result = execution.liquidate(
        [10.0],
        [0.25] * 4,
        spread=0.01,
        impact=[[0.5]],
        covariance=[[1.0]],
        aversion=aversion,
    )
    print(
        "Inventory aversion:",
        aversion,
        "trades:",
        result.trades[:, 0],
        "cash cost:",
        result.expected_cost,
        "risk preference:",
        result.inventory_penalty,
    )
rng = np.random.default_rng(3300)
x = np.column_stack([np.ones(1000), rng.normal(size=1000)])
true_beta = np.array([-0.4, 0.8])
probability = 1 / (1 + np.exp(-x @ true_beta))
trials = np.full(len(x), 10)
fit = execution.fit_assignment(x, rng.binomial(trials, probability), trials)
print("True/fitted assignment coefficients:", true_beta, fit.coefficients)
print(
    "Joint conditional assignment samples:",
    execution.assignment([3, 5], [0.2, 0.4], draws=5, rng=rng),
)
