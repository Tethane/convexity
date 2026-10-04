from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from numpy.typing import ArrayLike

from ..._typing import FloatArray
from .._backends import construction
from ..interfaces import GridProblem, MeshProblem, NumericalTool
from ..interfaces.base import run


@dataclass(frozen=True)
class GridConstruction:
    """Build a linear, logarithmic, geometric or arithmetic coordinate grid."""

    def __call__(self, problem: GridProblem):
        return construction.grid(problem)


@dataclass(frozen=True)
class MeshConstruction:
    """Combine coordinate axes with explicit mesh indexing or a Cartesian-product layout."""

    def __call__(self, problem: MeshProblem):
        return construction.mesh(problem)


def linspace(
    start: float,
    stop: float,
    num: int = 50,
    *,
    endpoint: bool = True,
    tool: NumericalTool[GridProblem, FloatArray] | None = None,
) -> FloatArray:
    """Return num evenly spaced scalar coordinates from start to stop.

    endpoint controls inclusion of stop; returns a read-only float64 array."""
    return run(
        GridProblem("linear", start, stop, num, endpoint=endpoint),
        tool if tool is not None else GridConstruction(),
    )


def logspace(
    start: float,
    stop: float,
    num: int = 50,
    *,
    base: float = 10.0,
    endpoint: bool = True,
    tool: NumericalTool[GridProblem, FloatArray] | None = None,
) -> FloatArray:
    """Uniformly spaced exponents, as in NumPy logspace."""
    return run(
        GridProblem("log", start, stop, num, endpoint=endpoint, base=base),
        tool if tool is not None else GridConstruction(),
    )


def geomspace(
    start: float,
    stop: float,
    num: int = 50,
    *,
    endpoint: bool = True,
    tool: NumericalTool[GridProblem, FloatArray] | None = None,
) -> FloatArray:
    """Return num geometrically spaced coordinates with the selected endpoint convention."""
    return run(
        GridProblem("geometric", start, stop, num, endpoint=endpoint),
        tool if tool is not None else GridConstruction(),
    )


def arange(
    start: float,
    stop: float,
    step: float = 1.0,
    *,
    tool: NumericalTool[GridProblem, FloatArray] | None = None,
) -> FloatArray:
    """Return a half-open arithmetic grid [start,stop) with nonzero step."""
    return run(
        GridProblem("range", start, stop, step=step),
        tool if tool is not None else GridConstruction(),
    )


def meshgrid(
    *axes: ArrayLike,
    sparse: bool = False,
    tool: NumericalTool[MeshProblem, tuple[FloatArray, ...]] | None = None,
) -> tuple[FloatArray, ...]:
    """Tensor mesh with indexing='ij'. Flattening uses C order throughout numerics."""
    return cast(
        tuple[FloatArray, ...],
        run(
            MeshProblem(axes, sparse=sparse),
            tool if tool is not None else MeshConstruction(),
        ),
    )


def cartesian_product(
    *axes: ArrayLike, tool: NumericalTool[MeshProblem, FloatArray] | None = None
) -> FloatArray:
    """Return all combinations of one-dimensional axes as rows, in input-axis order."""
    return cast(
        FloatArray,
        run(
            MeshProblem(axes, cartesian=True),
            tool if tool is not None else MeshConstruction(),
        ),
    )


__all__ = [
    "GridConstruction",
    "MeshConstruction",
    "linspace",
    "logspace",
    "geomspace",
    "arange",
    "meshgrid",
    "cartesian_product",
]
