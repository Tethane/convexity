import subprocess
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose

from convexity.math import distributions as dist, estimation as est


@pytest.mark.parametrize("initial,kwargs", [
    ([], {}), ([True], {}), (["1"], {}), ([1j], {}), ([np.nan], {}), ([np.inf], {}),
    ([[1.]], {}), ([0.], {"lower": 1.}), ([1.], {"lower": 1., "upper": 1.}),
    ([0.], {"lower": np.inf}), ([0.], {"upper": -np.inf}),
    ([0., 1.], {"names": ("a", "a")}), ([0.], {"names": ("a", "b")}),
    ([0., 1.], {"lower": [0., 0., 0.]}),
])
def test_invalid_parameter_spaces(initial, kwargs):
    with pytest.raises((TypeError, ValueError)):
        est.ParameterSpace(initial, **kwargs)


@pytest.mark.parametrize("weights", [[0, 0], [-1, 1], [.5, 1], [np.inf, 1], [1], [2**54, 1]])
def test_invalid_likelihood_frequency_weights(weights):
    with pytest.raises(ValueError):
        est.Likelihood([1, 2], lambda t, x: x, frequency_weights=weights)


@pytest.mark.parametrize("output", [1., [[1., 2.]], [np.nan, 1.], [np.inf, 1.], [1.], [True, False]])
def test_invalid_likelihood_contribution_shapes_and_values(output):
    model = est.Likelihood([1., 2.], lambda t, y: output)
    with pytest.raises(ValueError):
        model.log_likelihood([0.])


def test_ownership_and_read_only_callback_arguments():
    data = np.array([1., 2., 3.])
    initial = np.array([0.])
    model = est.likelihood(lambda t: dist.normal(t[0], 1), data)
    space = est.ParameterSpace(initial)
    data[:] = 9
    initial[:] = 7
    assert_allclose(np.asarray(model.data), [1., 2., 3.])
    assert_allclose(np.asarray(space.initial), [0.])

    def objective(theta):
        assert not theta.flags.writeable
        return (theta[0] - 2)**2

    result = est.minimize_estimate(objective, space)
    assert not result.parameters.flags.writeable
    with pytest.raises(ValueError):
        result.parameters[0] = 4
    with pytest.raises(TypeError):
        result.named_parameters["theta_0"] = 4  # type: ignore[index] -- test immutable mapping


def test_malformed_callbacks_are_rejected():
    with pytest.raises(ValueError, match="gradient"):
        est.minimize_estimate(lambda t: t @ t, [1.], gradient=lambda t: [1., 2.])
    with pytest.raises(ValueError, match="Jacobian"):
        est.least_squares(lambda t: [t[0]], [1.], jacobian=lambda t: [[1., 2.]])
    with pytest.raises(ValueError, match="finite"):
        est.minimize_estimate(lambda t: np.inf, [1.])
    with pytest.raises(ValueError, match="-inf"):
        est.minimize_estimate(lambda t: -np.inf, [1.])
    with pytest.raises(ValueError, match="constraint output"):
        est.minimize_estimate(lambda t: t @ t, [1.],
                               constraints=[est.NonlinearConstraint(lambda t: 1., lower=0, upper=1)])


@pytest.mark.parametrize("kwargs", [
    {"covariance": [[1.]]}, {"covariance_kind": "unknown"},
    {"covariance": [[-1.]], "covariance_kind": "test"},
    {"covariance": [[1., 0.], [0., 1.]], "covariance_kind": "test"},
    {"df": 0}, {"nobs": True},
])
def test_invalid_result_metadata(kwargs):
    with pytest.raises((TypeError, ValueError)):
        est.Estimate([1.], **kwargs)


def test_uncertainty_requires_correct_covariance_kind_and_convergence():
    point = est.Estimate([1.])
    with pytest.raises(ValueError):
        est.wald_interval(point)
    with pytest.raises(ValueError):
        est.delta_method(point, lambda t: t)
    point = est.Estimate([1.], covariance=[[1.]], covariance_kind="frequentist")
    with pytest.raises(ValueError):
        est.laplace_interval(point)
    for level in [0, 1, np.nan, -.5, True]:
        with pytest.raises(ValueError):
            est.wald_interval(point, level=level)
    failed = est.Estimate([1.], covariance=[[1.]], covariance_kind="frequentist",
                          diagnostics=est.Diagnostics(converged=False))
    with pytest.raises(est.ConvergenceError):
        est.wald_interval(failed)


def test_bootstrap_requires_explicit_rng_and_paired_rows():
    with pytest.raises(TypeError, match="Generator"):
        est.bootstrap(np.mean, ([1, 2, 3],), rng=42)
    with pytest.raises(ValueError, match="row counts"):
        est.bootstrap(lambda x, y: 0, ([1, 2], [1, 2, 3]), rng=np.random.default_rng(2))


def test_backend_cannot_claim_success_after_infinite_likelihood_trial():
    model = est.likelihood(lambda t: dist.bernoulli(t[0]), [0, 1, 1, 0, 1])
    space = est.ParameterSpace([.5], lower=0., upper=1.)
    # Some SciPy versions return the starting point with success=True after an
    # infinite endpoint trial. Accept a true solution or an honest failure.
    result = est.maximum_likelihood(model, space, estimator=est.ScipyMinimizer("L-BFGS-B"), check_convergence=False)
    if result.diagnostics.converged:
        assert_allclose(result.parameters, [.6], atol=2e-6)
    else:
        assert result.covariance is None


def test_import_does_not_load_other_families_or_backends():
    code = """
import sys
from convexity.math import estimation
assert 'convexity.math.numerical' not in sys.modules
assert 'convexity.math.distributions' not in sys.modules
assert 'torch' not in sys.modules
assert 'statsmodels' not in sys.modules
assert 'sklearn' not in sys.modules
assert 'estimation' in dir(__import__('convexity.math', fromlist=['']))
"""
    subprocess.run([sys.executable, "-c", code], check=True)
