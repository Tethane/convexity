"""negative wealth, options-sheet utility and preferences."""

import numpy as np
from convexity.math import utility

wealth = np.array([-5.0, 2.0, 10.0])
for model in (
    utility.LinearUtility(),
    utility.SmoothUtility(0.6, 3),
    utility.ExponentialUtility(0.1),
):
    print(
        type(model).__name__,
        "expected utility:",
        utility.expected_utility(wealth, model),
        "certainty equivalent:",
        utility.certainty_equivalent(wealth, model),
    )
print("Mean-variance:", utility.mean_variance(wealth, 0.1))
print("Mean-CVaR:", utility.mean_cvar(wealth, 0.2, 0.8))
print(
    "Noncash turnover/switch preference:",
    utility.preference_penalty([2.0, -1.0], turnover=0.2, switching=0.5),
)
