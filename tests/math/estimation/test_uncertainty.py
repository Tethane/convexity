import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from convexity.math import estimation as est


@pytest.mark.parametrize("method", ["percentile", "basic", "BCa"])
def test_bootstrap_matches_scipy_and_is_reproducible(method):
    data = np.array([1., 2., 3., 5., 8., 13.])
    result = est.bootstrap(np.mean, (data,), rng=np.random.default_rng(42), method=method, n_resamples=199)
    expected = stats.bootstrap((data,), np.mean, vectorized=False, rng=np.random.default_rng(42),
                               method=method, n_resamples=199)
    assert_allclose(result.interval.lower, [expected.confidence_interval.low])
    assert_allclose(result.interval.upper, [expected.confidence_interval.high])
    assert_allclose(result.draws[:, 0], expected.bootstrap_distribution)
    assert_allclose(result.estimate.standard_errors, np.atleast_1d(expected.standard_error))
    assert not result.draws.flags.writeable


def test_bootstrap_preserves_vector_rows_pairing_and_covariances():
    a = np.linspace(-2, 2, 40)
    x = np.column_stack((a, 3 * a + 2))
    y = 2 * a

    def statistic(design, response):
        assert_allclose(design[:, 1], 3 * design[:, 0] + 2)
        assert_allclose(response, 2 * design[:, 0])
        return [design[:, 0].mean(), response.mean()]

    result = est.bootstrap(statistic, (x, y), rng=np.random.default_rng(2), n_resamples=99, method="percentile")
    assert_allclose(result.draws[:, 1], 2 * result.draws[:, 0])
    assert result.estimate.covariance is not None
    variance = result.estimate.covariance[0, 0]
    assert_allclose(result.estimate.covariance, variance * np.array([[1, 2], [2, 4]]))


def test_bootstrap_of_estimator_and_no_discarded_failed_fits():
    x = np.column_stack((np.ones(30), np.arange(30)))
    y = x @ [1, 2] + np.sin(np.arange(30))
    result = est.bootstrap(lambda a, b: est.linear_least_squares(a, b, names=("a", "b")),
                           (x, y), rng=np.random.default_rng(3), n_resamples=49, method="basic")
    assert result.estimate.names == ("a", "b")
    assert result.draws.shape == (49, 2)
    calls = 0

    def sometimes_fails(sample):
        nonlocal calls
        calls += 1
        return est.Estimate([sample.mean()], diagnostics=est.Diagnostics(converged=calls == 1))

    with pytest.raises(est.ConvergenceError):
        est.bootstrap(sometimes_fails, (y,), rng=np.random.default_rng(5), method="percentile")


def test_degenerate_bca_is_not_silently_changed_to_another_method():
    with pytest.warns((RuntimeWarning, stats.DegenerateDataWarning)):
        with pytest.raises(ValueError, match="undefined"):
            est.bootstrap(np.mean, (np.ones(10),), rng=np.random.default_rng(3), n_resamples=19)


def test_delta_method_linear_identity_and_numerical_derivative():
    base = est.Estimate([1., 2.], covariance=[[2., .5], [.5, 3.]], covariance_kind="observed_information")
    a = np.array([[1., -1.], [2., 1.]])
    derived = est.delta_method(base, lambda t: a @ t, names=("difference", "combination"))
    assert derived.covariance is not None and base.covariance is not None
    assert_allclose(derived.parameters, a @ base.parameters)
    assert_allclose(derived.covariance, a @ base.covariance @ a.T)
    assert derived.df is None
    posterior = est.Estimate([0.], covariance=[[.2]], covariance_kind="laplace")
    transformed = est.delta_method(posterior, lambda t: np.exp(t[0]))
    assert transformed.covariance is not None
    assert_allclose(transformed.covariance, [[.2]], atol=1e-8)
    assert est.laplace_interval(transformed).kind == "credible"
