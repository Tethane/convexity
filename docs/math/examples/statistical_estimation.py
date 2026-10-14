"""Parameter recovery and uncertainty on simulated observations."""

import numpy as np
from numpy.testing import assert_allclose

from convexity.math import distributions as dist, estimation as est


def main():
    rng = np.random.default_rng(42)

    # A law is a model at one parameter point; a likelihood connects it to data.
    sample = dist.normal(2.0, 1.5).sample(500, rng=rng)
    model = est.likelihood(
        lambda theta: dist.normal(theta[0], np.exp(theta[1])), sample
    )
    result = est.maximum_likelihood(
        model, est.ParameterSpace([0.0, 0.0], names=("mu", "log_sigma"))
    )
    assert_allclose(result.parameters, [sample.mean(), np.log(sample.std())], atol=2e-5)
    sigma = est.delta_method(result, lambda theta: np.exp(theta[1]), names=("sigma",))
    print("Gaussian MLE:", dict(result.named_parameters), "sigma:", sigma.parameters)
    print("Mean/log-scale confidence intervals:", est.wald_interval(result))

    # Normal mean + normal prior: the optimizer agrees with conjugate algebra.
    known_scale = est.likelihood(lambda theta: dist.normal(theta[0], 1.5), sample[:12])
    posterior = est.maximum_a_posteriori(
        known_scale, est.gaussian_prior(0.0, 1.0), [0.0]
    )
    precision = 12 / 1.5**2 + 1
    assert_allclose(
        posterior.parameters, [sample[:12].sum() / 1.5**2 / precision], atol=1e-6
    )
    assert posterior.covariance is not None
    assert_allclose(posterior.covariance, [[1 / precision]], atol=1e-7)
    print(
        "MAP and approximate credible interval:",
        posterior.parameters,
        est.laplace_interval(posterior),
    )

    counts = dist.poisson(3.0).sample(200, rng=rng)
    poisson = est.maximum_likelihood(
        est.likelihood(lambda theta: dist.poisson(theta[0]), counts),
        est.ParameterSpace([1.0], names=("rate",), lower=0.0),
    )
    assert_allclose(poisson.parameters, [counts.mean()], atol=2e-5)
    print("Poisson MLE:", dict(poisson.named_parameters))

    # Fit a nonlinear residual model, then compare ordinary and robust estimates.
    t = np.linspace(0, 4, 80)
    y = 3 * np.exp(-0.7 * t) + rng.normal(0, 0.04, len(t))
    decay = est.least_squares(
        lambda theta: (theta[0] * np.exp(-theta[1] * t) - y) / 0.04,
        est.ParameterSpace([2.0, 0.5], names=("amplitude", "decay"), lower=0.0),
        absolute_sigma=True,
    )
    assert_allclose(decay.parameters, [3.0, 0.7], atol=0.04)
    print("Exponential decay:", dict(decay.named_parameters))

    x = np.linspace(-2, 2, 100)
    design = np.column_stack((np.ones(len(x)), x))
    response = design @ [1.0, 2.0] + rng.normal(0, 0.15, len(x))
    contaminated = response.copy()
    contaminated[-8:] += 15
    ordinary = est.linear_least_squares(design, contaminated)
    robust = est.robust_linear(design, contaminated)
    assert np.linalg.norm(robust.parameters - [1.0, 2.0]) < np.linalg.norm(
        ordinary.parameters - [1.0, 2.0]
    )
    print("Contaminated OLS / Huber:", ordinary.parameters, robust.parameters)
    print(
        "Ridge / lasso / elastic net:",
        est.ridge(design, response, alpha=0.1).parameters,
        est.lasso(design, response, alpha=0.1).parameters,
        est.elastic_net(design, response, alpha=0.1).parameters,
    )

    constrained = est.minimize_estimate(
        lambda theta: np.sum((theta - [0.8, 0.7]) ** 2),
        est.ParameterSpace([0.5, 0.5], lower=0.0, upper=1.0),
        constraints=[est.LinearConstraint([[1.0, 1.0]], lower=1.0, upper=1.0)],
    )
    assert_allclose(constrained.parameters, [0.55, 0.45], atol=1e-6)
    print("Constrained coefficients:", constrained.parameters)

    resampled = est.bootstrap(
        lambda a, b: est.linear_least_squares(a, b, names=("intercept", "slope")),
        (design, response),
        rng=rng,
        method="BCa",
        n_resamples=299,
    )
    assert resampled.draws.shape == (299, 2)
    print("Paired-row bootstrap intervals:", resampled.interval)


if __name__ == "__main__":
    main()
