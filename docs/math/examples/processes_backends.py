"""Optional QuantLib and sdeint integrations; install convexity[processes]."""

import numpy as np
from convexity.math import processes as p


def main():
    rng = np.random.default_rng(131)
    heston = p.Heston(0.05, 2.0, 0.04, 0.4, -0.7)
    paths = heston.sample(
        [100.0, 0.04], np.linspace(0, 1, 21), draws=8, rng=rng, substeps=4
    )
    print("Heston scheme/backend/exact:", paths.method, paths.backend, paths.exact)
    print("Heston terminal (spot,variance):", paths.values[:, -1])
    custom = p.ItoDiffusion(lambda t, x: -0.8 * x + 0.1 * np.sin(t), lambda t, x: [0.3])
    kernel = custom.transition(1.0, 1.0, method="sri2", substeps=64)
    print(
        "Custom Ito scheme/backend/exact:", kernel.method, kernel.backend, kernel.exact
    )
    print("Custom terminal samples:", kernel.sample(8, rng=rng))


if __name__ == "__main__":
    main()
