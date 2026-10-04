"""Private numpy execution for stopping."""

import numpy as np
from ..interfaces import StoppingResult


def snell(self, problem):
    p = problem
    values = np.array(p.rewards)
    continuation = np.zeros_like(values)
    region = np.zeros(values.shape, dtype=bool)
    region[-1] = True
    for t in range(len(values) - 2, -1, -1):
        continuation[t] = p.discounts[t] * (p.transitions[t] @ values[t + 1])
        region[t] = p.exercise[t] & (p.rewards[t] >= continuation[t])
        values[t] = np.where(region[t], p.rewards[t], continuation[t])
    return StoppingResult(values, continuation, region)
