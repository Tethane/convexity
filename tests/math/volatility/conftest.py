import numpy as np
import pytest


@pytest.fixture
def returns():
    rng = np.random.default_rng(421)
    y = np.empty(2500)
    h = 1.0
    previous = 0.0
    for t in range(len(y)):
        h = 0.08 + 0.12 * previous**2 + 0.80 * h
        previous = np.sqrt(h) * rng.normal()
        y[t] = previous
    return y[500:]


@pytest.fixture
def realized():
    rng = np.random.default_rng(17)
    z = np.zeros(1000)
    for t in range(22, len(z)):
        z[t] = (
            -0.10
            + 0.35 * z[t - 1]
            + 0.25 * z[t - 5 : t].mean()
            + 0.2 * z[t - 22 : t].mean()
            + rng.normal(scale=0.15)
        )
    return np.exp(z[200:])
