"""Ready-to-use linear algebra tools and convenience functions."""

from dataclasses import dataclass
from typing import Any, Literal

from numpy.typing import ArrayLike

from .. import _validation as v
from .._backends import linalg as backend
from ..interfaces import (
    ArrayResult, Eigenproblem, EigenResult, LinearSystem, MatrixExponential,
    NumericalTool, SVDProblem, SVDResult, run,
)


@dataclass(frozen=True)
class DenseSolve:
    """General dense square solve; singular/ill-conditioned systems raise."""

    def __call__(self, problem: LinearSystem) -> ArrayResult:
        return backend.dense_solve(problem)


@dataclass(frozen=True)
class SparseSolve:
    """Sparse direct, CG (requires positive definite A), or GMRES solve.

    Iterative methods accept one RHS. maxiter counts inner iterations for
    GMRES. No dense conversion or positive-definiteness factorization occurs.
    """

    method: Literal["direct", "cg", "gmres", "bicgstab"] = "direct"
    atol: float = 0.
    rtol: float = 1e-10
    maxiter: int = 1000
    preconditioner: Any = None

    def __post_init__(self) -> None:
        v.choice(self.method, ("direct", "cg", "gmres", "bicgstab"), "method")
        v.tolerances(self.atol, self.rtol)
        v.integer(self.maxiter, "maxiter")
        if self.preconditioner is not None:
            if self.method == "direct":
                raise ValueError("preconditioning applies only to iterative solvers")
            object.__setattr__(self, "preconditioner", v.matrix(self.preconditioner, "preconditioner"))

    def __call__(self, problem: LinearSystem) -> ArrayResult:
        return backend.sparse_solve(problem, method=self.method, atol=self.atol,
                                    rtol=self.rtol, maxiter=self.maxiter, preconditioner=self.preconditioner)


@dataclass(frozen=True)
class Eigendecomposition:
    def __call__(self, problem: Eigenproblem) -> EigenResult:
        return backend.eigen(problem)


@dataclass(frozen=True)
class SingularValueDecomposition:
    def __call__(self, problem: SVDProblem) -> SVDResult:
        return backend.svd(problem)


@dataclass(frozen=True)
class MatrixExponentialTool:
    def __call__(self, problem: MatrixExponential) -> ArrayResult:
        return backend.expm(problem)


def solve(a: Any, b: ArrayLike, *, tool: NumericalTool[LinearSystem, ArrayResult] | None = None,
          check_convergence: bool = True) -> ArrayResult:
    """Solve A x = b, selecting a dense or sparse direct solver by default."""
    if tool is None:
        tool = (SparseSolve(method="gmres") if backend.is_operator(a) else
                SparseSolve() if backend.is_sparse(a) else DenseSolve())
    return run(LinearSystem(a, b), tool, check_convergence=check_convergence)


def eig(a: ArrayLike, *, hermitian: bool = False,
        tool: NumericalTool[Eigenproblem, EigenResult] | None = None) -> EigenResult:
    """Right eigenpairs; Hermitian eigenvalues are ascending and real."""
    return run(Eigenproblem(a, hermitian), tool if tool is not None else Eigendecomposition())


def svd(a: ArrayLike, *, full_matrices: bool = False,
        tool: NumericalTool[SVDProblem, SVDResult] | None = None) -> SVDResult:
    """Reduced SVD by default; singular values are descending."""
    return run(SVDProblem(a, full_matrices), tool if tool is not None else SingularValueDecomposition())


def expm(a: ArrayLike, *, tool: NumericalTool[MatrixExponential, ArrayResult] | None = None) -> ArrayResult:
    """Dense matrix exponential, not elementwise exponentiation."""
    return run(MatrixExponential(a), tool if tool is not None else MatrixExponentialTool())


def sparse_matrix(data: Any, *, format: Literal["csr", "csc"] = "csr",
                  shape: tuple[int, int] | None = None) -> Any:
    """Create a native SciPy sparse array from dense, sparse, or COO-tuple data.

    COO form: (values, (rows, columns)), with shape=(m, n). Sparse arrays retain
    standard @, slicing, arithmetic, .toarray(), and .asformat() operations.
    """
    v.choice(format, ("csr", "csc"), "format")
    if shape is not None:
        if len(shape) != 2:
            raise ValueError("shape must have two dimensions")
        for size in shape:
            v.integer(size, "shape dimension")
    return backend.sparse_matrix(data, format=format, shape=shape)
