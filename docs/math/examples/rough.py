"""Generated rough paths, conditional forward curves, and measured lift error."""

import numpy as np
from convexity.math import rough as r


def main():
    times = np.linspace(0, 1, 33)
    rng = np.random.default_rng(140)
    fractional = r.FractionalBrownianMotion(0.15).sample(times, draws=3, rng=rng)
    print("fBm terminal values:", fractional.values[:, -1])

    initial = r.ForwardVarianceCurve([0, 0.5, 1, 2], [0.04, 0.045, 0.05, 0.055])
    model = r.RoughBergomi(hurst=0.15, eta=0.7, rho=-0.7, xi0=initial)
    result = model.sample(100, times, draws=3000, rng=rng)
    print("Terminal mean spot:", result.spot[:, -1].mean())
    print(
        "Terminal mean variance / initial forward variance:",
        result.variance[:, -1].mean(),
        initial(1.0),
    )
    print("Driver exact / spot exact:", result.driver.exact, result.paths.exact)

    index = 16
    forward = model.variance_model.forward_from_brownian(
        times[: index + 1],
        result.driver.brownian_increments[:3, :index],
        [0.5, 0.75, 1, 2],
    )
    print("Forward variances conditional on observed Brownian increments:")
    print(forward.values)

    for factors in (4, 8, 16, 32):
        _, error = r.approximate_fractional(0.15, factors=factors, horizon=1)
        print(
            f"{factors:2d} factors: relative L2 error={error.relative_l2_error:.6f}, "
            f"coupled terminal driver MSE={error.squared_error:.6f}"
        )
    kernel, error = r.approximate_fractional(
        0.15, factors=16, horizon=1, tolerance=0.15
    )
    lifted = r.VolterraBergomi(kernel, eta=0.7, rho=-0.7, xi0=initial)
    lifted_paths = lifted.sample(100, times, draws=3000, rng=rng, method="factors")
    assert lifted_paths.driver.factors is not None
    forward_lift = lifted.variance_model.forward_from_factors(
        0.5, lifted_paths.driver.factors[:3, index], [0.5, 0.75, 1, 2]
    )
    print("Full forward-variance states of the finite-factor model:")
    print(forward_lift.values)


if __name__ == "__main__":
    main()
