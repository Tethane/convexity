from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from numpy.typing import ArrayLike
from scipy.sparse import csr_array, csr_matrix

from .._backends import construction
from ..interfaces import DifferenceOperatorProblem, LaplacianProblem, NumericalTool
from ..interfaces.base import run


@dataclass(frozen=True)
class DifferenceOperator:
    """Build a finite-difference operator on the supplied coordinate axes and derivative orders."""

    def __call__(self, problem: DifferenceOperatorProblem):
        return construction.difference(problem)


@dataclass(frozen=True)
class LaplacianOperator:
    """Build a discrete Laplacian using the supplied grid shape, spacing and boundary convention."""

    def __call__(self, problem: LaplacianProblem):
        return construction.laplacian(problem)


def differentiation_matrix(
    x: ArrayLike,
    *,
    order: int = 1,
    accuracy: int = 2,
    periodic: bool = False,
    tool: NumericalTool[DifferenceOperatorProblem, csr_matrix | csr_array]
    | None = None,
) -> csr_matrix | csr_array:
    """Return a finite-difference matrix on increasing coordinates x.

    Multiplication by sampled values approximates the requested derivative order;
    accuracy controls the stencil. Periodic grids require the periodic convention."""
    return partial_derivative_matrix(
        (x,), (order,), accuracy=accuracy, periodic=periodic, tool=tool
    )


def partial_derivative_matrix(
    axes: Sequence[ArrayLike],
    orders: Sequence[int],
    *,
    accuracy: int = 2,
    periodic: bool = False,
    tool: NumericalTool[DifferenceOperatorProblem, csr_matrix | csr_array]
    | None = None,
) -> csr_matrix | csr_array:
    """Mixed partial derivative on a tensor grid; multiply by C-flattened samples.

    Nonperiodic boundaries use one-sided stencils, not imposed PDE conditions.
    Periodic axes omit the duplicate endpoint.
    """
    return run(
        DifferenceOperatorProblem(tuple(axes), tuple(orders), accuracy, periodic),
        tool if tool is not None else DifferenceOperator(),
    )


def laplacian_matrix(
    shape: tuple[int, ...],
    *,
    spacing: float = 1.0,
    boundary: str = "dirichlet",
    tool: NumericalTool[LaplacianProblem, csr_matrix | csr_array] | None = None,
) -> csr_matrix | csr_array:
    """Uniform isotropic grid Laplacian with homogeneous boundary conventions."""
    return run(
        LaplacianProblem(tuple(shape), spacing, boundary),
        tool if tool is not None else LaplacianOperator(),
    )


__all__ = [
    "DifferenceOperator",
    "LaplacianOperator",
    "differentiation_matrix",
    "partial_derivative_matrix",
    "laplacian_matrix",
]
