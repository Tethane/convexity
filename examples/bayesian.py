"""Exact updates and posterior prediction from generated observations."""
import numpy as np
from convexity.math import bayesian as b, distributions as d


def main():
    rng = np.random.default_rng(11)
    trials = np.full(30, 10)
    counts = rng.binomial(trials, .65)
    posterior = b.BetaBinomial(2, 2).update(counts, trials)
    print('Probability posterior mean:', posterior.posterior.mean())
    print('95% credible interval:', posterior.posterior.quantile([.025, .975]))
    print('Next 10 trials: mean, variance:', posterior.predictive(10).mean(), posterior.predictive(10).variance())

    observations = rng.normal(1.5, .8, 80)
    normal = b.NormalInverseGamma().update(observations)
    print('Joint (mean, variance) posterior:', normal.posterior.mean())
    print('Next observation interval:', normal.predictive().quantile([.025, .975]))
    simulated = b.posterior_predictive(normal.posterior,
        lambda theta: d.normal(np.asarray(theta)[0], np.sqrt(np.asarray(theta)[1])), draws=5000, rng=rng)
    print('Predictive variance, exact/Monte Carlo:', normal.predictive().variance(), simulated.variance())

    x = np.column_stack([np.ones(60), np.linspace(-1, 1, 60)])
    response = x@np.array([1., 2.])+rng.normal(0, .3, len(x))
    regression = b.GaussianLinearRegression([0, 0], 4*np.eye(2), .09).update(x, response)
    print('Regression coefficients:', regression.mean)
    print('Joint future covariance:', regression.predictive([[1, 1.1], [1, 1.2]]).covariance())


if __name__ == '__main__':
    main()
