from dataclasses import dataclass
from .._backends import construction
from ..interfaces.base import run
from ..interfaces.construction import GridProblem, MeshProblem


@dataclass(frozen=True)
class GridConstruction:
    def __call__(self, problem: GridProblem):
        return construction.grid(problem)


@dataclass(frozen=True)
class MeshConstruction:
    def __call__(self, problem: MeshProblem):
        return construction.mesh(problem)


def linspace(start, stop, num=50, *, endpoint=True, tool=None):
    return run(GridProblem("linear", start, stop, num, endpoint=endpoint), tool if tool is not None else GridConstruction())


def logspace(start, stop, num=50, *, base=10., endpoint=True, tool=None):
    """Uniformly spaced exponents, as in NumPy logspace."""
    return run(GridProblem("log", start, stop, num, endpoint=endpoint, base=base), tool if tool is not None else GridConstruction())


def geomspace(start, stop, num=50, *, endpoint=True, tool=None):
    return run(GridProblem("geometric", start, stop, num, endpoint=endpoint), tool if tool is not None else GridConstruction())


def arange(start, stop, step=1., *, tool=None):
    return run(GridProblem("range", start, stop, step=step), tool if tool is not None else GridConstruction())


def meshgrid(*axes, sparse=False, tool=None):
    """Tensor mesh with indexing='ij'. Flattening uses C order throughout numerics."""
    return run(MeshProblem(axes, sparse=sparse), tool if tool is not None else MeshConstruction())


def cartesian_product(*axes, tool=None):
    return run(MeshProblem(axes, cartesian=True), tool if tool is not None else MeshConstruction())


__all__ = ["GridConstruction", "MeshConstruction", "linspace", "logspace", "geomspace", "arange", "meshgrid", "cartesian_product"]
