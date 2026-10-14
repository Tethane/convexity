"""composed kernels, GP conditioning and correlated predictions."""

import numpy as np
from numpy.testing import assert_allclose

from convexity.math import gaussian_processes as gp


def main():
    rng = np.random.default_rng(42)
    x = np.linspace(-3, 3, 45)[:, None]
    y = np.sin(1.5 * x[:, 0]) + rng.normal(0, 0.12, len(x))
    query = np.linspace(-3.5, 3.5, 80)[:, None]
    kernel = gp.Constant(1.0, bounds=(0.05, 10)) * gp.Matern(
        1.0, nu=2.5, bounds=(0.1, 5)
    )
    fitted = gp.GaussianProcessRegressor(kernel, alpha=0.12**2, restarts=1, seed=7).fit(
        x, y
    )
    latent = fitted.posterior(query)
    observed = fitted.posterior(query, noise_variance=0.12**2)
    assert_allclose(observed.mean, latent.mean)
    assert_allclose(
        observed.covariance - latent.covariance,
        0.12**2 * np.eye(len(query)),
        atol=1e-12,
    )
    inside = np.abs(query[:, 0]) < 2.8
    assert np.mean((latent.mean[inside] - np.sin(1.5 * query[inside, 0])) ** 2) < 0.02
    draws = latent.sample(4, rng=np.random.default_rng(9))
    assert draws.shape == (4, len(query))
    assert latent.interval(0.95).shape == (len(query), 2)
    print("Fitted kernel:", fitted.kernel)
    print(
        "Training log marginal likelihood:", round(fitted.log_marginal_likelihood(), 4)
    )
    print("Optimizer:", fitted.diagnostics)
    print("Joint posterior draws:", draws.shape)

    # Learn observation noise as part of the covariance kernel instead.
    noisy_kernel = gp.Constant(1.0, bounds=(0.05, 10)) * gp.RBF(1.0, bounds=(0.1, 5))
    noisy_kernel += gp.WhiteNoise(0.02, bounds=(1e-4, 0.5))
    learned = gp.gaussian_process_regression(x, y, kernel=noisy_kernel)
    print("Kernel with fitted observation noise:", learned.kernel)
    # WhiteNoise is already included in this prediction's uncertainty.
    assert np.all(learned.predict_std(query) > 0)

    # Periodic covariance and an explicit prior use the same Gaussian result.
    periodic = gp.Periodic(0.8, 2.0, periodicity_bounds="fixed")
    assert_allclose(periodic(query, query + 2), periodic(query), atol=1e-13)
    prior = gp.GaussianPrediction(np.zeros(len(query)), periodic(query))
    assert prior.sample(2, rng=np.random.default_rng(3)).shape == (2, len(query))
    print("All family-7 examples passed.")


if __name__ == "__main__":
    main()
