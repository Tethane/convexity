"""Dense numerical linear algebra; least squares is an algebraic solve."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from numpy.typing import ArrayLike

from .. import _validation as v
from .._backends import linear_algebra as backend
from ..interfaces import (
    ArrayResult,
    ConditionResult,
    FactorizationProblem,
    FactorizationResult,
    GeneralizedEigenproblem,
    GeneralizedEigenResult,
    LeastSquaresProblem,
    LeastSquaresResult,
    LogDetResult,
    MatrixPropertyProblem,
    NumericalTool,
)
from ..interfaces.base import run


@dataclass(frozen=True)
class LeastSquares:
    """Solve linear least squares using a chosen numerical decomposition and rank threshold."""

    rcond: float | None = None

    def __post_init__(self) -> None:
        if self.rcond is not None:
            v.positive(self.rcond, "rcond", zero=True)

    def __call__(self, problem: LeastSquaresProblem):
        return backend.least_squares(problem, rcond=self.rcond)


@dataclass(frozen=True)
class GeneralizedEigen:
    """Solve A v = lambda B v, retaining homogeneous eigenvalue coordinates for singular B."""

    def __call__(self, problem: GeneralizedEigenproblem):
        return backend.generalized_eigen(problem)


@dataclass(frozen=True)
class Factorization:
    """Construct the requested QR, LU or Cholesky factors; inspect named factors and any permutation."""

    def __call__(self, problem: FactorizationProblem):
        return backend.factorize(problem)


@dataclass(frozen=True)
class MatrixProperties:
    """Evaluate a named matrix property such as rank, norm, condition or determinant."""

    def __call__(self, problem: MatrixPropertyProblem):
        return backend.matrix_property(problem)


def lstsq(
    a: ArrayLike,
    b: ArrayLike,
    *,
    tool: NumericalTool[LeastSquaresProblem, LeastSquaresResult] | None = None,
) -> LeastSquaresResult:
    """Minimize ||A @ x - b||_2 for A (m,n), b (m,) or (m,k).

    Return the minimum-norm solution, numerical rank and singular values; no
    statistical covariance is inferred. A user tool receives LeastSquaresProblem."""
    return run(LeastSquaresProblem(a, b), tool if tool is not None else LeastSquares())


def generalized_eig(
    a: ArrayLike,
    b: ArrayLike,
    *,
    hermitian: bool = False,
    tool: NumericalTool[GeneralizedEigenproblem, GeneralizedEigenResult] | None = None,
) -> GeneralizedEigenResult:
    """Solve A v = lambda B v for square matrices of equal shape.

    Return homogeneous alpha, beta and column eigenvectors: beta*A*v = alpha*B*v.
    Zero beta represents infinite eigenvalues; alpha=beta=0 is indeterminate."""
    return run(
        GeneralizedEigenproblem(a, b, hermitian),
        tool if tool is not None else GeneralizedEigen(),
    )


def qr(
    a: ArrayLike,
    *,
    pivoting: bool = False,
    tool: NumericalTool[FactorizationProblem, FactorizationResult] | None = None,
) -> FactorizationResult:
    """Factor A (m,n) into reduced Q and R.

    Return named factors and a column permutation p with A[:,p] = Q @ R.
    With pivoting=False, p is the identity."""
    return run(
        FactorizationProblem(a, "qr", pivoting),
        tool if tool is not None else Factorization(),
    )


def lu(
    a: ArrayLike,
    *,
    tool: NumericalTool[FactorizationProblem, FactorizationResult] | None = None,
) -> FactorizationResult:
    """Factor A (m,n) with row pivoting; return p, l, u satisfying A = p @ l @ u."""
    return run(
        FactorizationProblem(a, "lu"), tool if tool is not None else Factorization()
    )


def cholesky(
    a: ArrayLike,
    *,
    tool: NumericalTool[FactorizationProblem, FactorizationResult] | None = None,
) -> FactorizationResult:
    """Factor a Hermitian positive-definite matrix A; return l with A = l @ l.conj().T."""
    return run(
        FactorizationProblem(a, "cholesky"),
        tool if tool is not None else Factorization(),
    )


def schur(
    a: ArrayLike,
    *,
    tool: NumericalTool[FactorizationProblem, FactorizationResult] | None = None,
) -> FactorizationResult:
    """Return Schur factors t, z satisfying A = z @ t @ z.conj().T."""
    return run(
        FactorizationProblem(a, "schur"), tool if tool is not None else Factorization()
    )


def _property(a, operation, tool, *, order=None, tolerance=None):
    return run(
        MatrixPropertyProblem(a, operation, order, tolerance),
        tool if tool is not None else MatrixProperties(),
    )


def matrix_rank(
    a: ArrayLike,
    *,
    tolerance: float | None = None,
    tool: NumericalTool[MatrixPropertyProblem, ArrayResult] | None = None,
) -> ArrayResult:
    """Return the numerical rank of A (m,n), using an optional absolute singular-value tolerance."""
    return cast(ArrayResult, _property(a, "rank", tool, tolerance=tolerance))


def condition_number(
    a: ArrayLike,
    *,
    order: float | str | None = None,
    tool: NumericalTool[MatrixPropertyProblem, ConditionResult] | None = None,
) -> ConditionResult:
    """Return ||A|| * ||A^-1|| in the selected norm; singular matrices may return infinity."""
    return cast(ConditionResult, _property(a, "condition", tool, order=order))


def norm(
    a: ArrayLike,
    *,
    order: float | str | None = None,
    tool: NumericalTool[MatrixPropertyProblem, ArrayResult] | None = None,
) -> ArrayResult:
    """Return a scalar vector/matrix norm; order follows NumPy norm conventions."""
    return cast(ArrayResult, _property(a, "norm", tool, order=order))


def pinv(
    a: ArrayLike,
    *,
    tolerance: float | None = None,
    tool: NumericalTool[MatrixPropertyProblem, ArrayResult] | None = None,
) -> ArrayResult:
    """Return the Moore-Penrose pseudoinverse of A (m,n), shaped (n,m).

    Tolerance controls numerical rank, not an elementwise approximation error."""
    return cast(ArrayResult, _property(a, "pinv", tool, tolerance=tolerance))


def det(
    a: ArrayLike,
    *,
    tool: NumericalTool[MatrixPropertyProblem, ArrayResult] | None = None,
) -> ArrayResult:
    """Return the scalar determinant of a square real or complex matrix."""
    return cast(ArrayResult, _property(a, "det", tool))


def slogdet(
    a: ArrayLike,
    *,
    tool: NumericalTool[MatrixPropertyProblem, LogDetResult] | None = None,
) -> LogDetResult:
    """Return determinant phase/sign and log absolute magnitude.

    A singular determinant has sign=0 and logabsdet=-inf; complex phase is preserved."""
    return cast(LogDetResult, _property(a, "slogdet", tool))


def logm(
    a: ArrayLike,
    *,
    tool: NumericalTool[MatrixPropertyProblem, ArrayResult] | None = None,
) -> ArrayResult:
    """Return the principal matrix logarithm, which may be complex for real input."""
    return cast(ArrayResult, _property(a, "logm", tool))


def sqrtm(
    a: ArrayLike,
    *,
    tool: NumericalTool[MatrixPropertyProblem, ArrayResult] | None = None,
) -> ArrayResult:
    """Return a principal matrix square root, which may be complex for real input."""
    return cast(ArrayResult, _property(a, "sqrtm", tool))


__all__ = [
    "LeastSquares",
    "GeneralizedEigen",
    "Factorization",
    "MatrixProperties",
    "lstsq",
    "generalized_eig",
    "qr",
    "lu",
    "cholesky",
    "schur",
    "matrix_rank",
    "condition_number",
    "norm",
    "pinv",
    "det",
    "slogdet",
    "logm",
    "sqrtm",
]
