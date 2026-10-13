"""Family 22: inspect the exact finite-state Snell envelope."""

import numpy as np
from convexity.math.stopping import StoppingProblem, BackwardInduction

rewards = [[1.0, 1.0], [3.0, 0.0], [0.0, 6.0]]
transitions = np.tile([[0.5, 0.5], [0.5, 0.5]], (2, 1, 1))
result = BackwardInduction().solve(StoppingProblem(rewards, transitions, 0.9))
print("Values:", result.values)
print("Continuation:", result.continuation)
print("Exercise:", result.exercise_region)
