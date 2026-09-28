"""Concrete numerical tools backed by mature libraries."""

from .calculus import AdaptiveQuadrature, AutoDiff, FiniteDifference, derivative, gradient, hessian, jacobian, quad
from .interpolation import Interpolation, interpolate
from .linalg import (
    DenseSolve, Eigendecomposition, MatrixExponentialTool, SingularValueDecomposition,
    SparseSolve, eig, eigh, expm, solve, sparse_matrix, svd,
)
from .matrix_equations import LyapunovSolver, RiccatiSolver, solve_lyapunov, solve_riccati
from .roots import ScalarRootSolver, SystemRootSolver, root, root_scalar


from .linear_algebra import (
    LeastSquares,
    GeneralizedEigen,
    Factorization,
    MatrixProperties,
    lstsq,
    generalized_eig,
    qr,
    lu,
    cholesky,
    schur,
    matrix_rank,
    condition_number,
    norm,
    pinv,
    det,
    slogdet,
    logm,
    sqrtm,
)

from .sparse import (
    MatrixFreeOperator,
    SparseSpectral,
    linear_operator,
    sparse_eig,
    sparse_svd,
)

from .fixed_point import (
    FixedPointSolver,
    fixed_point,
)

from .products import (
    FiniteDifferenceProduct,
    AutoDiffProduct,
    jvp,
    vjp,
    check_derivative,
)

from .integrate import (
    MultidimensionalQuadrature,
    GaussianQuadrature,
    GaussianRule,
    nquad,
    gauss_quad,
    quadrature_rule,
    integrate_density,
)

from .approximation import (
    Spline,
    MultivariateInterpolation,
    PolynomialApproximation,
    spline,
    regular_grid,
    scattered,
    polynomial_approximation,
)

from .ode import (
    IVPSolver,
    BVPSolver,
    solve_ivp,
    solve_ode,
    solve_bvp,
)

from .grids import (
    GridConstruction,
    MeshConstruction,
    linspace,
    logspace,
    geomspace,
    arange,
    meshgrid,
    cartesian_product,
)

from .operators import (
    DifferenceOperator,
    LaplacianOperator,
    differentiation_matrix,
    partial_derivative_matrix,
    laplacian_matrix,
)

from .accuracy import (
    FloatingComparison,
    Residual,
    isclose,
    allclose,
    machine_precision,
    require_finite,
    residual,
)

from .backends import (
    scipy_backend,
    torch_backend,
)

__all__ = [
    'AdaptiveQuadrature',
    'AutoDiff',
    'FiniteDifference',
    'derivative',
    'gradient',
    'hessian',
    'jacobian',
    'quad',
    'Interpolation',
    'interpolate',
    'DenseSolve',
    'Eigendecomposition',
    'MatrixExponentialTool',
    'SingularValueDecomposition',
    'SparseSolve',
    'eig',
    'eigh',
    'expm',
    'solve',
    'sparse_matrix',
    'svd',
    'LyapunovSolver',
    'RiccatiSolver',
    'solve_lyapunov',
    'solve_riccati',
    'ScalarRootSolver',
    'SystemRootSolver',
    'root',
    'root_scalar',
    'LeastSquares',
    'GeneralizedEigen',
    'Factorization',
    'MatrixProperties',
    'lstsq',
    'generalized_eig',
    'qr',
    'lu',
    'cholesky',
    'schur',
    'matrix_rank',
    'condition_number',
    'norm',
    'pinv',
    'det',
    'slogdet',
    'logm',
    'sqrtm',
    'MatrixFreeOperator',
    'SparseSpectral',
    'linear_operator',
    'sparse_eig',
    'sparse_svd',
    'FixedPointSolver',
    'fixed_point',
    'FiniteDifferenceProduct',
    'AutoDiffProduct',
    'jvp',
    'vjp',
    'check_derivative',
    'MultidimensionalQuadrature',
    'GaussianQuadrature',
    'GaussianRule',
    'nquad',
    'gauss_quad',
    'quadrature_rule',
    'integrate_density',
    'Spline',
    'MultivariateInterpolation',
    'PolynomialApproximation',
    'spline',
    'regular_grid',
    'scattered',
    'polynomial_approximation',
    'IVPSolver',
    'BVPSolver',
    'solve_ivp',
    'solve_ode',
    'solve_bvp',
    'GridConstruction',
    'MeshConstruction',
    'linspace',
    'logspace',
    'geomspace',
    'arange',
    'meshgrid',
    'cartesian_product',
    'DifferenceOperator',
    'LaplacianOperator',
    'differentiation_matrix',
    'partial_derivative_matrix',
    'laplacian_matrix',
    'FloatingComparison',
    'Residual',
    'isclose',
    'allclose',
    'machine_precision',
    'require_finite',
    'residual',
    'scipy_backend',
    'torch_backend',
]
