"""Covariance identities, composition, gradients and backend extension."""

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from convexity.math import gaussian_processes as gp


def test_kernel_formulas_and_cross_covariances():
    x = np.array([[0., 0.], [1., 2.], [-2., 1.]])
    y = np.array([[.5, 1.], [1., -1.]])
    lengths = np.array([.5, 2.])
    r2 = np.sum(((x[:, None] - y) / lengths) ** 2, axis=-1)
    assert_allclose(gp.RBF(lengths)(x, y), np.exp(-r2 / 2))
    r = np.sqrt(r2)
    assert_allclose(gp.Matern(lengths, nu=.5)(x, y), np.exp(-r))
    assert_allclose(gp.Matern(lengths, nu=1.5)(x, y), (1 + np.sqrt(3)*r)*np.exp(-np.sqrt(3)*r))
    assert_allclose(gp.Matern(lengths, nu=np.inf)(x, y), gp.RBF(lengths)(x, y))
    assert_allclose(gp.DotProduct(.3)(x, y), x @ y.T + .09)
    assert_allclose(gp.Constant(2)(x, y), np.full((3, 2), 2.))
    r2 = np.sum((x[:, None] - y)**2, axis=-1)
    assert_allclose(gp.RationalQuadratic(2., .7)(x, y), (1 + r2 / (2*.7*4))**(-.7))


def test_periodic_identity_and_dimension_contract_in_composition():
    x = np.linspace(-2, 2, 12)[:, None]
    kernel = gp.Periodic(.8, 1.3)
    assert_allclose(kernel(x, x + 1.3), kernel(x), atol=1e-14)
    assert np.linalg.eigvalsh(kernel(x)).min() > -1e-12
    for k in (kernel, 2 * kernel + gp.RBF()):
        with pytest.raises(ValueError, match="one-dimensional"):
            k(np.ones((3, 2)))
        with pytest.raises(ValueError, match="one-dimensional"):
            gp.GaussianProcessRegressor(k).fit(np.ones((3, 2)), [1, 2, 3])


def test_white_noise_is_per_observation_not_coordinate_equality():
    x = np.array([[1.], [1.], [2.]])
    k = gp.WhiteNoise(.2)
    assert_allclose(k(x), .2*np.eye(3))
    assert_allclose(k(x, x), np.zeros((3, 3)))
    assert_allclose(k.diag(x), [.2]*3)


def test_composition_and_fixed_scalar_arithmetic():
    x = np.linspace(-1, 1, 9)[:, None]
    a, b = gp.RBF(.7), gp.DotProduct(.5)
    k = (2*a + b + .3)**2
    assert_allclose(k(x), (2*a(x) + b(x) + .3)**2)
    assert_allclose(k.diag(x), np.diag(k(x)))
    assert np.linalg.eigvalsh(k(x)).min() > -1e-12
    assert k.theta.size == 2  # Scalar operands introduce no fitted parameters.
    summed: Any = sum([a, b])
    assert_allclose((summed * 0)(x), np.zeros((9, 9)))


@pytest.mark.parametrize("kernel", [gp.RBF([.7, 1.2]), gp.Matern([1., 2.], nu=2.5),
                                     gp.RationalQuadratic(.8, .4), gp.WhiteNoise(.3),
                                     (gp.Constant(2)*gp.RBF(.7) + gp.DotProduct(.5))**2])
def test_log_parameter_gradients_by_finite_differences(kernel):
    x = np.random.default_rng(8).normal(size=(6, 2))
    matrix, gradient = kernel(x, eval_gradient=True)
    assert gradient.shape == (6, 6, len(kernel.theta))
    for j in range(len(kernel.theta)):
        step = np.eye(len(kernel.theta))[j] * 1e-6
        numerical = (kernel.with_theta(kernel.theta + step)(x) -
                     kernel.with_theta(kernel.theta - step)(x)) / 2e-6
        assert_allclose(gradient[..., j], numerical, atol=1e-6, rtol=1e-5)
    assert_allclose(matrix, kernel(x))
    assert len(kernel.parameter_names) == len(kernel.theta)


def test_fixed_parameters_and_immutable_updates():
    k = gp.RBF(.5, bounds="fixed") + gp.WhiteNoise(.1, bounds="fixed")
    assert k.theta.shape == (0,)
    assert k.bounds.shape == (0, 2)
    assert k([[1.], [2.]], eval_gradient=True)[1].shape == (2, 2, 0)
    free = gp.RBF(.5)
    moved = free.with_theta([np.log(2.)])
    assert_allclose(free.theta, [np.log(.5)])
    assert_allclose(moved.theta, [np.log(2.)])


def test_custom_sklearn_kernel_is_copied():
    from sklearn.gaussian_process.kernels import RBF

    class Custom(RBF):
        pass

    backend = Custom(.7)
    kernel = gp.from_sklearn(backend)
    backend.length_scale = 9
    assert_allclose(kernel.theta, [np.log(.7)])
    x = np.linspace(0, 1, 8)[:, None]
    model = gp.GaussianProcessRegressor(kernel, optimize=False).fit(x, np.sin(x[:, 0]))
    assert_allclose(model.predict(x), np.sin(x[:, 0]), atol=1e-5)
