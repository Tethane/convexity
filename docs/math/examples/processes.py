"""Exact stochastic transitions, generators, and generated irregular paths."""

import numpy as np
from convexity.math import processes as p, distributions as d


def main():
    rng = np.random.default_rng(130)
    ou = p.OrnsteinUhlenbeck(1.8, 0.5, 0.4)
    kernel = ou.transition(-1.0, 0.75)
    print("OU conditional mean/variance:", kernel.mean(), kernel.covariance())
    print(
        "OU conditional quantiles:", kernel.distribution().quantile([0.05, 0.5, 0.95])
    )
    print("OU generator of x² at x=1:", ou.generator(0, 1.0, 2.0, 2.0))
    paths = ou.sample(-1.0, [0.0, 0.1, 0.4, 1.0, 2.0], draws=5, rng=rng)
    print("Exact joint OU paths:", paths.values)

    vector = p.VectorOrnsteinUhlenbeck(
        [[2.0, -0.3], [0.2, 1.0]], [1.0, -1.0], [[0.4, 0.1], [0.1, 0.3]]
    )
    print("Vector OU stationary covariance:", vector.stationary().covariance())
    cir = p.CIR(2.0, 0.04, 0.5)
    print("CIR Feller condition:", cir.feller_satisfied)
    print(
        "CIR exact law from boundary:",
        cir.transition(0.0, 0.5).distribution().quantile([0.05, 0.5, 0.95]),
    )
    print(
        "GBM expected level:",
        p.GeometricBrownianMotion(0.05, 0.2).transition(100.0, 1.0).mean(),
    )

    jumps = p.AdditiveJumpDiffusion(0.1, [0.2], 0.8, d.normal(-0.3, 0.1))
    print(
        "Additive jump mean/variance:",
        jumps.transition(0.0, 1.0).mean(),
        jumps.transition(0.0, 1.0).covariance(),
    )
    merton = p.MertonJumpDiffusion(0.05, 0.2, 0.8, -0.1, 0.2).transition(100.0, 1.0)
    samples = merton.sample(20000, rng=rng)
    print("Merton analytic/sample expected levels:", merton.mean(), samples.mean())


if __name__ == "__main__":
    main()
