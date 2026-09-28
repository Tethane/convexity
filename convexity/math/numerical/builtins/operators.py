from dataclasses import dataclass
from .._backends import construction
from ..interfaces.base import run
from ..interfaces.construction import DifferenceOperatorProblem, LaplacianProblem


@dataclass(frozen=True)
class DifferenceOperator:
    def __call__(self, problem: DifferenceOperatorProblem):
        return construction.difference(problem)


@dataclass(frozen=True)
class LaplacianOperator:
    def __call__(self, problem: LaplacianProblem):
        return construction.laplacian(problem)


def differentiation_matrix(x, *, order=1, accuracy=2, periodic=False, tool=None):
    return partial_derivative_matrix((x,), (order,), accuracy=accuracy, periodic=periodic, tool=tool)


def partial_derivative_matrix(axes, orders, *, accuracy=2, periodic=False, tool=None):
    """Mixed partial derivative on a tensor grid; multiply by C-flattened samples.

    Nonperiodic boundaries use one-sided stencils, not imposed PDE conditions.
    Periodic axes omit the duplicate endpoint.
    """
    return run(DifferenceOperatorProblem(tuple(axes), tuple(orders), accuracy, periodic),
               tool if tool is not None else DifferenceOperator())


def laplacian_matrix(shape, *, spacing=1., boundary="dirichlet", tool=None):
    """Uniform isotropic grid Laplacian with homogeneous boundary conventions."""
    return run(LaplacianProblem(tuple(shape), spacing, boundary), tool if tool is not None else LaplacianOperator())


__all__ = ["DifferenceOperator", "LaplacianOperator", "differentiation_matrix", "partial_derivative_matrix", "laplacian_matrix"]
