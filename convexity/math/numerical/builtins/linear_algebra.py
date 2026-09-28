"""Dense numerical linear algebra; least squares is an algebraic solve."""
from dataclasses import dataclass
from typing import cast

from .. import _validation as v
from .._backends import linear_algebra as backend
from ..interfaces.base import ArrayResult, run
from ..interfaces.linear_algebra import (
    FactorizationProblem, GeneralizedEigenproblem, LeastSquaresProblem, MatrixPropertyProblem,
    ConditionResult, LogDetResult,
)


@dataclass(frozen=True)
class LeastSquares:
    rcond: float | None = None

    def __post_init__(self):
        if self.rcond is not None:
            v.positive(self.rcond, "rcond", zero=True)

    def __call__(self, problem: LeastSquaresProblem):
        return backend.least_squares(problem, rcond=self.rcond)


@dataclass(frozen=True)
class GeneralizedEigen:
    def __call__(self, problem: GeneralizedEigenproblem):
        return backend.generalized_eigen(problem)


@dataclass(frozen=True)
class Factorization:
    def __call__(self, problem: FactorizationProblem):
        return backend.factorize(problem)


@dataclass(frozen=True)
class MatrixProperties:
    def __call__(self, problem: MatrixPropertyProblem):
        return backend.matrix_property(problem)


def lstsq(a, b, *, tool=None):
    return run(LeastSquaresProblem(a, b), tool if tool is not None else LeastSquares())


def generalized_eig(a, b, *, hermitian=False, tool=None):
    return run(GeneralizedEigenproblem(a, b, hermitian), tool if tool is not None else GeneralizedEigen())


def qr(a, *, pivoting=False, tool=None):
    return run(FactorizationProblem(a, "qr", pivoting), tool if tool is not None else Factorization())


def lu(a, *, tool=None):
    return run(FactorizationProblem(a, "lu"), tool if tool is not None else Factorization())


def cholesky(a, *, tool=None):
    return run(FactorizationProblem(a, "cholesky"), tool if tool is not None else Factorization())


def schur(a, *, tool=None):
    return run(FactorizationProblem(a, "schur"), tool if tool is not None else Factorization())


def _property(a, operation, tool, *, order=None, tolerance=None):
    return run(MatrixPropertyProblem(a, operation, order, tolerance), tool if tool is not None else MatrixProperties())


def matrix_rank(a, *, tolerance=None, tool=None) -> ArrayResult:
    return cast(ArrayResult, _property(a, "rank", tool, tolerance=tolerance))


def condition_number(a, *, order=None, tool=None) -> ConditionResult:
    return cast(ConditionResult, _property(a, "condition", tool, order=order))


def norm(a, *, order=None, tool=None) -> ArrayResult:
    return cast(ArrayResult, _property(a, "norm", tool, order=order))


def pinv(a, *, tolerance=None, tool=None) -> ArrayResult:
    return cast(ArrayResult, _property(a, "pinv", tool, tolerance=tolerance))


def det(a, *, tool=None) -> ArrayResult:
    return cast(ArrayResult, _property(a, "det", tool))


def slogdet(a, *, tool=None) -> LogDetResult:
    return cast(LogDetResult, _property(a, "slogdet", tool))


def logm(a, *, tool=None) -> ArrayResult:
    return cast(ArrayResult, _property(a, "logm", tool))


def sqrtm(a, *, tool=None) -> ArrayResult:
    return cast(ArrayResult, _property(a, "sqrtm", tool))


__all__ = ["LeastSquares", "GeneralizedEigen", "Factorization", "MatrixProperties", "lstsq",
           "generalized_eig", "qr", "lu", "cholesky", "schur", "matrix_rank", "condition_number",
           "norm", "pinv", "det", "slogdet", "logm", "sqrtm"]
