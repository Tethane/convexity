"""Changing measurement units must not make an invalid covariance valid."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from convexity.math import (
    bayesian,
    distributions,
    estimation,
    gaussian_processes,
    state_space,
)


def estimate(mean, covariance):
    return estimation.Estimate(mean, covariance=covariance, covariance_kind="sampling")


def student_t(mean, covariance):
    return distributions.multivariate_student_t(5, mean, covariance)


FACTORIES = [
    distributions.multivariate_normal,
    student_t,
    estimate,
    gaussian_processes.GaussianPrediction,
    state_space.GaussianState,
    bayesian.GaussianLinearRegression,
]


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("scale", [1.0, 1e-20, 1e20])
def test_material_asymmetry_is_rejected_in_any_units(factory, scale):
    with pytest.raises(ValueError, match="symmetric"):
        factory([0.0, 0.0], scale * np.array([[2.0, 0.6], [0.2, 1.0]]))


@pytest.mark.parametrize(
    "factory",
    [estimate, gaussian_processes.GaussianPrediction, state_space.GaussianState],
)
@pytest.mark.parametrize("scale", [1.0, 1e-20, 1e20])
def test_indefinite_covariance_is_rejected_in_any_units(factory, scale):
    # Positive diagonal alone is insufficient: this matrix has eigenvalue -scale.
    with pytest.raises(ValueError, match="positive semidefinite"):
        factory([0.0, 0.0], scale * np.array([[1.0, 2.0], [2.0, 1.0]]))


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("scale", [1e-20, 1.0, 1e20])
def test_valid_covariance_preserves_units(factory, scale):
    covariance = scale * np.array([[2.0, 0.6], [0.6, 1.0]])
    result = factory([0.0, 0.0], covariance)
    if isinstance(result, distributions.Distribution):
        actual = result.covariance()
        # The Student-t's input matrix is a scale, not its covariance.
        if factory is student_t:
            actual = actual * 3 / 5
    else:
        actual = result.covariance
    assert_allclose(actual / scale, covariance / scale, rtol=1e-13, atol=1e-13)
