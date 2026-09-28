"""Family-5 extension boundaries, input validation, ownership and lazy imports."""

import subprocess
import sys
from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from convexity.math import dependence as dep


@pytest.mark.parametrize("bad", [[], [1, 2], [[True]], [["1"]], [[1j]], [[np.nan]], [[np.inf]], np.empty((2, 0))])
@pytest.mark.parametrize("fitter", [dep.SampleCovariance(), dep.EWMACovariance(), dep.LedoitWolf(),
                                    dep.OAS(), dep.ShrunkCovariance(), dep.StudentT(), dep.KendallCopula()])
def test_training_contract(bad, fitter):
    with pytest.raises((TypeError, ValueError)):
        fitter.fit(bad)


@pytest.mark.parametrize("constructor,kwargs", [
    (dep.SampleCovariance, {"ddof": True}), (dep.SampleCovariance, {"ddof": 2}),
    (dep.SampleCovariance, {"assume_centered": 1}), (dep.EWMACovariance, {"decay": 0}),
    (dep.EWMACovariance, {"decay": 1.01}), (dep.EWMACovariance, {"decay": np.nan}),
    (dep.ShrunkCovariance, {"shrinkage": -1}), (dep.LedoitWolf, {"assume_centered": 2}),
    (dep.OAS, {"assume_centered": "yes"}), (dep.PositiveSemidefinite, {"min_eigenvalue": -1}),
    (dep.PositiveSemidefinite, {"tolerance": -1}), (dep.IndependenceCopula, {"dimension": 1}),
    (dep.GaussianCopula, {"correlation": [[1, 1], [1, 1]]}),
    (dep.StudentTCopula, {"correlation": np.eye(2), "df": 0}),
    (dep.ClaytonCopula, {"theta": 0}), (dep.GumbelCopula, {"theta": .9}),
    (dep.KendallCopula, {"family": "student_t"}), (dep.KendallCopula, {"df": 4}),
    (dep.KendallCopula, {"family": "unknown"}), (dep.KendallCopula, {"repair_correlation": 1}),
    (dep.StudentT, {"df": -1}), (dep.StudentT, {"df_bounds": (10, 1)}),
    (dep.StudentT, {"estimate_df": True, "df": 300}), (dep.StudentT, {"maxiter": 0}),
    (dep.StudentT, {"tolerance": 0}), (dep.GeneralizedPareto, {"maxiter": True}),
    (dep.GeneralizedExtremeValue, {"check_convergence": 1}),
])
def test_invalid_specifications(constructor, kwargs):
    with pytest.raises((TypeError, ValueError)):
        constructor(**kwargs)


def test_copied_readonly_state_and_outputs():
    location = np.array([1., 2.])
    covariance = np.array([[1., .3], [.3, 1.]])
    result = dep.CovarianceEstimate(location, covariance)
    copula: Any = dep.GaussianCopula(covariance)
    factor = dep.FactorCovariance([[1.], [2.]], [[2.]], [1., 1.])
    location[:] = 99
    covariance[:] = 0
    assert_allclose(result.location, [1, 2])
    assert_allclose(copula.correlation, result.covariance)
    for array in (result.location, result.covariance, result.precision, result.correlation,
                  result.standard_deviation, result.variance([1, 1]), copula.correlation,
                  copula.sample(3, rng=np.random.default_rng(1)), factor.loadings,
                  factor.factor_covariance, factor.residual_covariance):
        assert not array.flags.writeable
    with pytest.raises(FrozenInstanceError):
        copula.adapter = None
    bounds: Any = [.5, 100.]
    specification = dep.StudentT(df_bounds=bounds)
    bounds[0] = 10
    assert specification.df_bounds == (.5, 100.)
    decay: Any = np.array(.9)
    ewma = dep.EWMACovariance(decay=decay)
    decay[...] = .2
    assert ewma.decay == .9


def test_user_covariance_estimator_and_invalid_return_states():
    class Custom:
        def fit(self, samples):
            assert not samples.flags.writeable
            return dep.CovarianceEstimate(samples.mean(axis=0), np.eye(samples.shape[1]), nobs=len(samples))

    x = np.arange(12.).reshape(6, 2)
    result = dep.fit_covariance(x, Custom())
    assert_allclose(result.covariance, np.eye(2))

    class Bad:
        def fit(self, samples):
            return dep.CovarianceEstimate([0], [[1]])

    with pytest.raises(ValueError):
        dep.fit_covariance(x, Bad())


def test_user_copula_adapter_capabilities_and_result_checks():
    class Custom:
        dimension = 2
        capabilities = frozenset({"cdf", "log_density"})

        def evaluate(self, operation, *args, **kwargs):
            assert not args[0].flags.writeable
            if operation == "cdf":
                return np.prod(args[0], axis=-1)
            return np.zeros(args[0].shape[:-1])

    result = dep.Copula(Custom())
    assert_allclose(result.cdf([[.5, .5], [.2, .8]]), [.25, .16])
    with pytest.raises(NotImplementedError):
        result.sample(2, rng=np.random.default_rng(1))

    class Bad(Custom):
        def evaluate(self, operation, *args, **kwargs):
            return np.array([2.])

    with pytest.raises(ValueError):
        dep.Copula(Bad()).cdf([.5, .5])
    with pytest.raises(ValueError, match="leading"):
        dep.Copula(Bad()).log_density([.5, .5])


def test_invalid_compositions_and_results():
    for construct in (
        lambda: dep.FactorCovariance(np.eye(2), [[1]], [1, 1]),
        lambda: dep.FactorCovariance(np.eye(2), np.eye(2), [-1, 1]),
        lambda: dep.CovarianceEstimate([0, 0], [[1]]),
        lambda: dep.CovarianceEstimate([0], [[1]], nobs=3, effective_nobs=4),
        lambda: dep.fit_factor_covariance(np.ones((5, 2)), np.ones((5, 1))),
        lambda: dep.fit_factor_covariance(np.ones((5, 2)), np.ones((4, 1))),
        lambda: dep.nearest_correlation(np.eye(2)*2),
        lambda: dep.correlation_to_covariance(np.eye(2), [-1, 1]),
        lambda: dep.correlation([[1, 2], [1, 3]]),
        lambda: dep.generalized_pareto(.2, scale=0),
        lambda: dep.generalized_extreme_value(np.nan),
    ):
        with pytest.raises((TypeError, ValueError)):
            construct()


def test_import_isolation_and_exports():
    code = """
import sys
from convexity.math import dependence
for name in ('convexity.math.distributions', 'convexity.math.estimation',
             'convexity.math.transforms', 'scipy', 'sklearn', 'statsmodels', 'torch', 'pandas'):
    assert name not in sys.modules, name
for name in dependence.__all__:
    assert hasattr(dependence, name), name
assert 'dependence' in dir(__import__('convexity.math', fromlist=['']))
"""
    subprocess.run([sys.executable, "-c", code], check=True)
