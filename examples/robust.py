"""Family 30: robust mean ambiguity, CVaR and finite-support KL ambiguity."""

import numpy as np
from convexity.math import robust

mean = np.array([0.05, 0.09])
covariance = np.array([[0.01, 0.002], [0.002, 0.04]])
for radius in (0.0, 0.2, 1.0):
    result = robust.robust_allocation(
        mean,
        covariance,
        robust.Ellipsoid(mean, [[0.01, 0], [0, 0.05]], radius),
        aversion=2.0,
    )
    print("Ambiguity radius:", radius, "allocation:", result.solution)
returns = [[0.1, -0.2], [-0.1, 0.2], [0.05, 0.08]]
print("Mean-CVaR allocation:", robust.mean_cvar(returns, level=0.8).solution)
for radius in (0.0, 0.1, 1.0):
    result = robust.kl_worst_case([0.0, 2.0, 10.0], radius, weights=[0.7, 0.25, 0.05])
    print(
        "KL radius:",
        radius,
        "worst loss:",
        result.objective,
        "adversarial masses:",
        result.solution,
    )
