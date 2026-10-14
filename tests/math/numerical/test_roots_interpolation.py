import numpy as np
import pytest
from numpy.testing import assert_allclose

from convexity.math import numerical as nm


@pytest.mark.parametrize("method", ["brentq", "bisect", "toms748", "newton", "secant"])
def test_scalar_root_methods(method):
    result = nm.root_scalar(
        lambda x: x * x - 2,
        bracket=(0, 2),
        x0=1,
        x1=2,
        derivative=lambda x: 2 * x,
        tool=nm.ScalarRootSolver(method=method),
    )
    assert_allclose(result.value, np.sqrt(2), atol=1e-9)
    assert result.diagnostics.residual_norm is not None
    assert result.diagnostics.residual_norm < 1e-8


@pytest.mark.parametrize("method", ["brentq", "bisect", "toms748"])
@pytest.mark.parametrize("endpoint", [0, 1])
def test_endpoint_root_diagnostics(method, endpoint):
    result = nm.root_scalar(
        lambda x: x - endpoint, bracket=(0, 1), tool=nm.ScalarRootSolver(method=method)
    )
    assert result.value == endpoint
    assert result.diagnostics.iterations == 0
    assert result.diagnostics.residual_norm == 0


def test_root_failures_are_not_silently_successful():
    with pytest.raises(ValueError, match="change sign"):
        nm.root_scalar(lambda x: x * x + 1, bracket=(-1, 1))
    with pytest.raises(nm.ConvergenceError) as caught:
        nm.root_scalar(
            lambda x: x * x - 2,
            bracket=(0, 2),
            tool=nm.ScalarRootSolver(method="bisect", maxiter=1),
        )
    assert isinstance(caught.value.result, nm.ArrayResult)
    assert not caught.value.result.diagnostics.converged
    result = nm.root(lambda x: [x[0] ** 2 + 1], [0.0], check_convergence=False)
    assert not result.diagnostics.converged
    assert result.diagnostics.residual_norm is not None
    assert result.diagnostics.residual_norm >= 1


def test_square_system_with_analytic_jacobian():
    result = nm.root(
        lambda x: [x[0] ** 2 + x[1] - 3, x[0] + x[1] ** 2 - 3],
        [1.0, 1.0],
        jacobian=lambda x: [[2 * x[0], 1], [1, 2 * x[1]]],
    )
    assert_allclose(result.value, np.full(2, (np.sqrt(13) - 1) / 2), atol=1e-10)
    assert result.diagnostics.residual_norm is not None
    assert result.diagnostics.residual_norm < 1e-10


@pytest.mark.parametrize(
    "factory",
    [
        lambda: nm.root_scalar(lambda x: x, bracket=(1, 0)),
        lambda: nm.root_scalar(lambda x: x, x0=1),
        lambda: nm.root_scalar(
            lambda x: x, x0=1, tool=nm.ScalarRootSolver(method="newton")
        ),
        lambda: nm.root_scalar(
            lambda x: x, x0=1, x1=1, tool=nm.ScalarRootSolver(method="secant")
        ),
        lambda: nm.root(lambda x: [1, 2], [1]),
        lambda: nm.root(lambda x: x, [1], jacobian=lambda x: [1]),
    ],
)
def test_root_input_contracts(factory):
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize("method", ["linear", "cubic", "pchip"])
def test_interpolation_knots_and_multichannel_shapes(method):
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.stack([x, x * x], axis=1)
    fitted = nm.interpolate(x, y, tool=nm.Interpolation(method))
    assert_allclose(fitted(x), y, atol=1e-13)
    assert fitted(0.5).shape == (2,)
    assert fitted([[0.5, 1.5], [2.5, 3.0]]).shape == (2, 2, 2)
    with pytest.raises(ValueError, match="outside"):
        fitted(-0.1)


def test_interpolation_cubic_polynomial_linear_extrapolation_and_pchip_monotonicity():
    x = np.arange(5, dtype=float)
    points = np.linspace(0, 4, 101)
    assert_allclose(
        nm.interpolate(x, x**3, tool=nm.Interpolation("cubic"))(points),
        points**3,
        atol=1e-12,
    )
    assert_allclose(nm.interpolate([0, 1], [1, 3], extrapolate=True)([-1, 2]), [-1, 5])
    fitted = nm.interpolate(
        x, [0.0, 1.0, 1.1, 2.0, 10.0], tool=nm.Interpolation("pchip")
    )
    assert np.all(np.diff(fitted(points)) >= 0)


@pytest.mark.parametrize(
    "x,y",
    [
        ([0, 0], [1, 2]),
        ([1, 0], [1, 2]),
        ([0], [1]),
        ([0, 1], [1]),
        ([0, np.nan], [1, 2]),
        ([0, 1], [1, np.inf]),
    ],
)
def test_invalid_interpolation_data(x, y):
    with pytest.raises(ValueError):
        nm.interpolate(x, y)
