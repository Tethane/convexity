import numpy as np
import pytest
from numpy.testing import assert_allclose
from convexity.math import numerical as n


def test_multidimensional_box_and_improper_integration():
    result = n.nquad(lambda x: x[0] ** 2 + x[1], [[0, 1], [0, 2]])
    assert_allclose(result.value, 8 / 3, atol=1e-10)
    result = n.nquad(lambda x: np.exp(-x[0] - x[1]), [[0, np.inf], [0, np.inf]])
    assert_allclose(result.value, 1.0, atol=1e-8)


def test_gaussian_quadrature_polynomial_exactness_and_density_integral():
    for degree in range(8):
        assert_allclose(
            n.gauss_quad(lambda x: x**degree, -1, 1, n=4).value,
            0 if degree % 2 else 2 / (degree + 1),
            atol=1e-14,
        )
    assert_allclose(n.gauss_quad(lambda x: x * x, 1, 0).value, -1 / 3)
    assert_allclose(
        n.integrate_density(lambda x: x * x, lambda x: np.exp(-x), 0, np.inf).value,
        2.0,
        atol=1e-9,
    )
    with pytest.raises(ValueError, match="nonnegative"):
        n.integrate_density(lambda x: x, lambda x: -1, 0, 1)


@pytest.mark.parametrize(
    "family,weight_mass",
    [("legendre", 2), ("hermite", np.sqrt(np.pi)), ("laguerre", 1)],
)
def test_quadrature_nodes_weights(family, weight_mass):
    rule = n.quadrature_rule(12, family=family)
    assert np.all(np.diff(rule.nodes) > 0)
    assert np.all(rule.weights > 0)
    assert_allclose(np.sum(rule.weights), weight_mass, atol=1e-13)
    if family == "hermite":
        assert_allclose(rule.weights @ rule.nodes**2, np.sqrt(np.pi) / 2)
    if family == "laguerre":
        assert_allclose(rule.weights @ rule.nodes**2, 2)


def test_multidimensional_quadrature_failure_and_bad_shapes():
    with pytest.raises(n.ConvergenceError):
        n.nquad(
            lambda x: np.sin(1000 * x[0]),
            [[0, 1]],
            tool=n.MultidimensionalQuadrature(limit=1),
        )
    with pytest.raises(ValueError):
        n.nquad(lambda x: 1, [[0, 1, 2]])
    with pytest.raises(ValueError):
        n.gauss_quad(np.exp, 0, np.inf)


@pytest.mark.parametrize("method", ["linear", "nearest", "cubic"])
def test_regular_grid_scalar_multichannel_and_bounds(method):
    x, y = np.linspace(0, 1, 5), np.linspace(-1, 1, 6)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    values = np.stack([xx + 2 * yy, 3 * xx - yy], axis=-1)
    fitted = n.regular_grid((x, y), values, tool=n.MultivariateInterpolation(method))
    assert_allclose(fitted([x[2], y[3]]), values[2, 3], atol=1e-8)
    if method != "nearest":
        assert_allclose(
            fitted([[0.15, -0.22], [0.74, 0.12]]),
            [[-0.29, 0.67], [0.98, 2.1]],
            atol=1e-8,
        )
    assert fitted(np.zeros((2, 3, 2))).shape == (2, 3, 2)
    with pytest.raises(ValueError):
        fitted([1.1, 0])


def test_regular_grid_extrapolation_policies():
    axes, values = ([0, 1], [0, 1]), np.array([[0, 1], [1, 2]])
    assert np.isnan(n.regular_grid(axes, values, extrapolation="nan")([2, 2]))
    assert_allclose(
        n.regular_grid(axes, values, extrapolation="extrapolate")([2, 2]), 4
    )


@pytest.mark.parametrize("method", ["linear", "nearest", "cubic"])
def test_scattered_hull_policy(method):
    points = np.array([[0, 0], [1, 0], [0, 1], [0.25, 0.25]])
    values = points[:, 0] + 2 * points[:, 1]
    tool = n.MultivariateInterpolation(method)
    fitted = n.scattered(points, values, tool=tool)
    assert_allclose(fitted(points), values, atol=1e-10)
    if method != "nearest":
        assert_allclose(fitted([0.2, 0.3]), 0.8, atol=1e-6)
    # Inside bounding rectangle but outside convex hull.
    with pytest.raises(ValueError, match="hull"):
        fitted([0.9, 0.9])
    assert np.isnan(
        n.scattered(points, values, extrapolation="nan", tool=tool)([0.9, 0.9])
    )


def test_scattered_nearest_extrapolation_and_invalid_linear_policy():
    points, values = [[0, 0], [1, 0], [0, 1]], [0, 1, 2]
    fitted = n.scattered(
        points,
        values,
        extrapolation="extrapolate",
        tool=n.MultivariateInterpolation("nearest"),
    )
    assert fitted([2, 0]) == 1
    with pytest.raises(ValueError, match="nearest"):
        n.scattered(points, values, extrapolation="extrapolate")


@pytest.mark.parametrize("basis", ["polynomial", "chebyshev"])
def test_polynomial_function_approximation(basis):
    x = np.linspace(-2, 3, 20)
    fitted = n.polynomial_approximation(
        x, 1 + 2 * x - 3 * x * x, 2, tool=n.PolynomialApproximation(basis)
    )
    assert_allclose(fitted.coefficients, [1, 2, -3], atol=1e-12)
    assert_allclose(
        fitted([0.1, 0.2]),
        1 + 2 * np.array([0.1, 0.2]) - 3 * np.array([0.1, 0.2]) ** 2,
        atol=1e-12,
    )
    assert fitted.rank == 3
    with pytest.raises(ValueError):
        fitted(4)


@pytest.mark.parametrize("degree", [1, 3, 5])
def test_splines(degree):
    x = np.linspace(0, 1, 8)
    fitted = n.spline(x, x**degree, tool=n.Spline(degree))
    assert_allclose(
        fitted(np.linspace(0, 1, 21)), np.linspace(0, 1, 21) ** degree, atol=1e-12
    )


def test_cubic_spline_natural_and_clamped_boundaries():
    x = np.linspace(0, 1, 5)
    natural = n.spline(x, x, tool=n.Spline(boundary="natural"))
    assert_allclose(natural([0.1, 0.7]), [0.1, 0.7], atol=1e-13)
    clamped = n.spline(x, x * x * (3 - 2 * x), tool=n.Spline(boundary="clamped"))
    assert_allclose(
        clamped([0.1, 0.7]),
        np.array([0.1, 0.7]) ** 2 * (3 - 2 * np.array([0.1, 0.7])),
        atol=1e-12,
    )
