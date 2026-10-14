"""Independent conditioning checks and sklearn parity on important conventions."""

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from convexity.math import gaussian_processes as gp


def test_posterior_and_likelihood_match_gaussian_conditioning():
    x = np.array([[-1.0], [0.0], [1.0]])
    y = np.array([0.2, -0.4, 1.2])
    query = np.array([[-0.5], [0.7], [2.0]])
    noise = np.array([0.02, 0.04, 0.08])
    kernel = 1.4 * gp.RBF(0.6)
    model = gp.GaussianProcessRegressor(kernel, alpha=noise, optimize=False).fit(x, y)
    training = kernel(x) + np.diag(noise)
    cross = kernel(query, x)
    mean = cross @ np.linalg.solve(training, y)
    covariance = kernel(query) - cross @ np.linalg.solve(training, cross.T)
    law = model.posterior(query)
    assert_allclose(law.mean, mean, atol=1e-14)
    assert_allclose(law.covariance, covariance, atol=1e-14)
    assert_allclose(model.predict(query), mean)
    assert_allclose(model.predict_std(query), np.sqrt(np.diag(covariance)))
    expected_lml = -0.5 * (
        y @ np.linalg.solve(training, y)
        + np.linalg.slogdet(training)[1]
        + len(x) * np.log(2 * np.pi)
    )
    assert_allclose(model.log_marginal_likelihood(), expected_lml)
    assert model.nobs == 3 and model.input_dimension == 1
    assert not model.diagnostics.optimized


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("constant_target", [False, True])
def test_normalization_and_white_noise_match_sklearn(normalize, constant_target):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

    x = np.linspace(-1, 1, 10)[:, None]
    y = np.full(10, 5.0) if constant_target else 5 + 3 * np.sin(2 * x[:, 0])
    q = np.array([[-0.7], [0.4], [0.4], [1.6]])
    options: dict[str, Any] = dict(alpha=0.07, optimizer=None, normalize_y=normalize)
    backend = GaussianProcessRegressor(
        ConstantKernel(2) * RBF(0.6) + WhiteKernel(0.03), **options
    ).fit(x, y)
    model = gp.GaussianProcessRegressor(
        gp.Constant(2) * gp.RBF(0.6) + gp.WhiteNoise(0.03),
        alpha=0.07,
        optimize=False,
        normalize_y=normalize,
    ).fit(x, y)
    mean, cov = backend.predict(q, return_cov=True)
    law = model.posterior(q)
    assert_allclose(law.mean, mean, atol=1e-12)
    assert_allclose(law.covariance, cov, atol=1e-12)
    observed = model.posterior(q, noise_variance=[0.1, 0.2, 0.3, 0.4])
    assert_allclose(
        observed.covariance - law.covariance, np.diag([0.1, 0.2, 0.3, 0.4]), atol=1e-12
    )
    assert_allclose(
        model.predict_std(q, noise_variance=0.3), np.sqrt(np.diag(cov) + 0.3)
    )


def test_alpha_and_white_noise_have_distinct_predictive_variances():
    x = np.linspace(-1, 1, 8)[:, None]
    y = np.sin(x[:, 0])
    q = np.array([[0.2], [0.2], [0.8]])
    latent = (
        gp.GaussianProcessRegressor(gp.RBF(), alpha=0.2, optimize=False)
        .fit(x, y)
        .posterior(q)
    )
    noisy = (
        gp.GaussianProcessRegressor(
            gp.RBF() + gp.WhiteNoise(0.2), alpha=0, optimize=False
        )
        .fit(x, y)
        .posterior(q)
    )
    assert_allclose(noisy.mean, latent.mean)
    assert_allclose(noisy.covariance - latent.covariance, 0.2 * np.eye(3), atol=1e-14)
    assert_allclose(latent.covariance[0], latent.covariance[1], atol=1e-14)


def test_interpolation_and_singular_training_inputs():
    x = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.4, 1.2, -0.5])
    model = gp.GaussianProcessRegressor(gp.RBF(0.5), alpha=0, optimize=False).fit(x, y)
    law = model.posterior(x)
    assert_allclose(law.mean, y, atol=1e-14)
    assert_allclose(law.covariance, 0, atol=1e-14)
    with pytest.raises(np.linalg.LinAlgError):
        gp.GaussianProcessRegressor(gp.RBF(), alpha=0, optimize=False).fit(
            [[0], [0]], [1, 1]
        )
    regularized = gp.GaussianProcessRegressor(gp.RBF(), alpha=1e-5, optimize=False).fit(
        [[0], [0]], [1, 1]
    )
    assert np.isfinite(regularized.predict([[0.0]])).all()


def test_likelihood_gradient_and_evaluation_do_not_change_fit():
    x = np.linspace(0, 2, 15)[:, None]
    model = gp.GaussianProcessRegressor(
        gp.Constant(1.2) * gp.RBF(0.7), alpha=0.03, optimize=False
    ).fit(x, np.cos(x[:, 0]))
    theta = model.kernel.theta
    before = model.predict(x)
    value, gradient = model.log_marginal_likelihood(eval_gradient=True)
    assert value == model.log_marginal_likelihood()
    for j in range(len(theta)):
        step = np.eye(len(theta))[j] * 1e-6
        numerical = (
            model.log_marginal_likelihood(theta + step)
            - model.log_marginal_likelihood(theta - step)
        ) / 2e-6
        assert_allclose(gradient[j], numerical, rtol=1e-5, atol=1e-5)
    assert_allclose(model.predict(x), before, atol=0, rtol=0)
    assert_allclose(model.kernel.theta, theta, atol=0, rtol=0)


def test_hyperparameter_fitting_improves_likelihood_and_restarts_are_reproducible():
    rng = np.random.default_rng(12)
    x = np.linspace(-2, 2, 35)[:, None]
    y = np.sin(2 * x[:, 0]) + rng.normal(0, 0.08, len(x))
    kernel = gp.Constant(0.4, bounds=(0.05, 5)) * gp.RBF(0.15, bounds=(0.05, 3))
    initial = gp.GaussianProcessRegressor(kernel, alpha=0.08**2, optimize=False).fit(
        x, y
    )
    spec = gp.GaussianProcessRegressor(kernel, alpha=0.08**2, restarts=1, seed=9)
    a, b = spec.fit(x, y), spec.fit(x, y)
    assert a.log_marginal_likelihood() > initial.log_marginal_likelihood() + 10
    assert_allclose(a.kernel.theta, b.kernel.theta, atol=1e-12)
    assert_allclose(kernel.theta, initial.kernel.theta)
    assert a.diagnostics.converged and a.diagnostics.optimizer_runs == 2
    q = np.linspace(-1.9, 1.9, 40)[:, None]
    assert np.mean((a.predict(q) - np.sin(2 * q[:, 0])) ** 2) < 0.01


def test_optimizer_failure_retains_result_and_bound_warning_is_not_failure(monkeypatch):
    from scipy.optimize import OptimizeResult

    def failed(fun, x0, **kwargs):
        return OptimizeResult(
            x=x0, fun=fun(x0)[0], success=False, nit=1, message="iteration limit"
        )

    monkeypatch.setattr("scipy.optimize.minimize", failed)
    x = np.linspace(0, 1, 6)[:, None]
    y = np.sin(x[:, 0])
    with pytest.raises(gp.ConvergenceError, match="iteration limit") as error:
        gp.GaussianProcessRegressor(gp.RBF()).fit(x, y)
    assert not error.value.result.diagnostics.converged
    assert np.isfinite(error.value.result.predict(x)).all()
    unchecked = gp.GaussianProcessRegressor(gp.RBF(), check_convergence=False).fit(x, y)
    assert not unchecked.diagnostics.converged
    fixed = gp.GaussianProcessRegressor(gp.RBF(bounds="fixed")).fit(x, y)
    assert fixed.diagnostics.converged and fixed.diagnostics.optimizer_runs == 0


def test_bound_warnings_are_reported_without_rejecting_successful_fit():
    x = np.linspace(0, 1, 8)[:, None]
    model = gp.GaussianProcessRegressor(gp.RBF(0.2, bounds=(0.1, 0.3)), alpha=0.01).fit(
        x, np.ones(8)
    )
    assert model.diagnostics.converged
    assert any("bound" in warning for warning in model.diagnostics.warnings)
