"""Public linalg namespace; implementation adapters remain private."""
from .builtins.linalg import (
    DenseSolve,
    SparseSolve,
    Eigendecomposition,
    SingularValueDecomposition,
    MatrixExponentialTool,
    solve,
    eig,
    svd,
    expm,
    sparse_matrix,
)
from .builtins.linear_algebra import (
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
from .builtins.sparse import (
    MatrixFreeOperator,
    SparseSpectral,
    linear_operator,
    sparse_eig,
    sparse_svd,
)
from .builtins.matrix_equations import (
    LyapunovSolver,
    RiccatiSolver,
    solve_lyapunov,
    solve_riccati,
)

def eigh(a, b=None, *, tool=None):
    return eig(a, hermitian=True, tool=tool) if b is None else generalized_eig(a, b, hermitian=True, tool=tool)

__all__ = ['DenseSolve', 'SparseSolve', 'Eigendecomposition', 'SingularValueDecomposition', 'MatrixExponentialTool', 'solve', 'eig', 'svd', 'expm', 'sparse_matrix', 'LeastSquares', 'GeneralizedEigen', 'Factorization', 'MatrixProperties', 'lstsq', 'generalized_eig', 'qr', 'lu', 'cholesky', 'schur', 'matrix_rank', 'condition_number', 'norm', 'pinv', 'det', 'slogdet', 'logm', 'sqrtm', 'MatrixFreeOperator', 'SparseSpectral', 'linear_operator', 'sparse_eig', 'sparse_svd', 'LyapunovSolver', 'RiccatiSolver', 'solve_lyapunov', 'solve_riccati', 'eigh']
