"""Built-in capability bundles; custom callables can replace any capability."""

from __future__ import annotations

from typing import Any

from .._backends.linalg import is_operator, is_sparse
from ..interfaces.backend import Backend
from ..interfaces.base import NumericalTool
from .accuracy import FloatingComparison, Residual
from .approximation import MultivariateInterpolation, PolynomialApproximation, Spline
from .calculus import AdaptiveQuadrature, AutoDiff, FiniteDifference
from .fixed_point import FixedPointSolver
from .grids import GridConstruction, MeshConstruction
from .integrate import GaussianQuadrature, GaussianRule, MultidimensionalQuadrature
from .interpolation import Interpolation
from .linalg import (
    DenseSolve,
    Eigendecomposition,
    MatrixExponentialTool,
    SingularValueDecomposition,
    SparseSolve,
)
from .linear_algebra import (
    Factorization,
    GeneralizedEigen,
    LeastSquares,
    MatrixProperties,
)
from .matrix_equations import LyapunovSolver, RiccatiSolver
from .ode import BVPSolver, IVPSolver
from .operators import DifferenceOperator, LaplacianOperator
from .products import AutoDiffProduct, FiniteDifferenceProduct
from .roots import ScalarRootSolver, SystemRootSolver
from .sparse import MatrixFreeOperator, SparseSpectral


class _LinearSolve:
    def __call__(self, problem):
        tool = (
            SparseSolve(method="gmres")
            if is_operator(problem.a)
            else SparseSolve()
            if is_sparse(problem.a)
            else DenseSolve()
        )
        return tool(problem)


def scipy_backend() -> Backend:
    """NumPy/SciPy numerical tools, with findiff for differentiation matrices."""
    tools: dict[str, NumericalTool[Any, Any]] = dict(
        solve=_LinearSolve(),
        eig=Eigendecomposition(),
        svd=SingularValueDecomposition(),
        expm=MatrixExponentialTool(),
        lstsq=LeastSquares(),
        generalized_eig=GeneralizedEigen(),
        sparse_eig=SparseSpectral(),
        sparse_svd=SparseSpectral(),
        linear_operator=MatrixFreeOperator(),
        quad=AdaptiveQuadrature(),
        nquad=MultidimensionalQuadrature(),
        gauss_quad=GaussianQuadrature(),
        quadrature_rule=GaussianRule(),
        root_scalar=ScalarRootSolver(),
        root=SystemRootSolver(),
        fixed_point=FixedPointSolver(),
        interpolate=Interpolation(),
        spline=Spline(),
        regular_grid=MultivariateInterpolation(),
        scattered=MultivariateInterpolation(),
        polynomial_approximation=PolynomialApproximation(),
        solve_ivp=IVPSolver(),
        solve_bvp=BVPSolver(),
        solve_lyapunov=LyapunovSolver(),
        solve_riccati=RiccatiSolver(),
        differentiation_matrix=DifferenceOperator(),
        partial_derivative_matrix=DifferenceOperator(),
        laplacian_matrix=LaplacianOperator(),
        isclose=FloatingComparison(),
        residual=Residual(),
    )
    tools.update({name: Factorization() for name in ("qr", "lu", "cholesky", "schur")})
    tools.update(
        {
            name: MatrixProperties()
            for name in (
                "matrix_rank",
                "condition_number",
                "norm",
                "pinv",
                "det",
                "slogdet",
                "logm",
                "sqrtm",
            )
        }
    )
    tools.update(
        {
            name: FiniteDifference()
            for name in ("derivative", "gradient", "jacobian", "hessian")
        }
    )
    tools.update({name: FiniteDifferenceProduct() for name in ("jvp", "vjp")})
    tools.update(
        {
            name: GridConstruction()
            for name in ("linspace", "logspace", "geomspace", "arange")
        }
    )
    tools.update(
        {name: MeshConstruction() for name in ("meshgrid", "cartesian_product")}
    )
    return Backend("numpy-scipy", tools)


def torch_backend(*, device: str = "cpu") -> Backend:
    """Autodiff capabilities only; unsupported operations fail explicitly."""
    tools = {
        name: AutoDiff(device=device)
        for name in ("derivative", "gradient", "jacobian", "hessian")
    }
    return Backend(
        "pytorch",
        {**tools, "jvp": AutoDiffProduct(device), "vjp": AutoDiffProduct(device)},
    )


__all__ = ["scipy_backend", "torch_backend"]
