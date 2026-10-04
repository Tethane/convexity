from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csr_array, csr_matrix

from ..._typing import FloatArray
from .. import _validation as v


@dataclass(frozen=True)
class GridProblem:
    """Coordinate-grid specification; kind determines whether num, step or base controls construction."""

    kind: str
    start: float
    stop: float
    num: int = 50
    step: float = 1.0
    endpoint: bool = True
    base: float = 10.0

    def __post_init__(self) -> None:
        v.choice(self.kind, ("linear", "log", "geometric", "range"), "kind")
        v.scalar(self.start, "start")
        v.scalar(self.stop, "stop")
        v.integer(self.num, "num")
        v.boolean(self.endpoint, "endpoint")
        v.positive(self.base, "base")
        if self.kind == "log" and self.base == 1:
            raise ValueError("logarithmic base must differ from one")
        if self.kind == "geometric" and (
            self.start == 0
            or self.stop == 0
            or np.sign(self.start) != np.sign(self.stop)
        ):
            raise ValueError(
                "geometric endpoints must be nonzero and have the same sign"
            )
        if self.kind == "range":
            step = v.scalar(self.step, "step")
            if step == 0 or (self.stop - self.start) * step <= 0:
                raise ValueError("range step must point from start toward stop")

    def validate_result(self, result: object) -> None:
        if (
            not isinstance(result, np.ndarray)
            or result.ndim != 1
            or not result.size
            or not np.all(np.isfinite(result))
        ):
            raise ValueError("grid tool must return a nonempty finite vector")
        if self.kind != "range" and result.size != self.num:
            raise ValueError("grid tool returned the wrong number of points")


@dataclass(frozen=True)
class MeshProblem:
    """Coordinate axes and indexing convention for a tensor mesh or Cartesian product."""

    axes: tuple[FloatArray, ...]
    cartesian: bool = False
    sparse: bool = False

    def __post_init__(self) -> None:
        v.boolean(self.cartesian, "cartesian")
        v.boolean(self.sparse, "sparse")
        if not self.axes:
            raise ValueError("supply at least one axis")
        if self.cartesian and self.sparse:
            raise ValueError("Cartesian products cannot use sparse mesh storage")
        object.__setattr__(
            self,
            "axes",
            tuple(v.array(x, "axis", ndim=1, real=True) for x in self.axes),
        )

    def validate_result(self, result: object) -> None:
        shape = tuple(x.size for x in self.axes)
        if self.cartesian:
            expected = (int(np.prod(shape)), len(shape))
            if (
                not isinstance(result, np.ndarray)
                or result.shape != expected
                or not np.all(np.isfinite(result))
            ):
                raise ValueError("invalid Cartesian product shape/values")
        else:
            if not isinstance(result, tuple) or len(result) != len(shape):
                raise ValueError("mesh must contain one array per dimension")
            for i, component in enumerate(result):
                expected = (
                    tuple(n if i == j else 1 for j, n in enumerate(shape))
                    if self.sparse
                    else shape
                )
                if (
                    not isinstance(component, np.ndarray)
                    or component.shape != expected
                    or not np.all(np.isfinite(component))
                ):
                    raise ValueError("invalid mesh component shape/values")

    def __init__(
        self,
        axes: tuple[ArrayLike, ...],
        cartesian: bool = False,
        sparse: bool = False,
    ) -> None:
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "cartesian", cartesian)
        object.__setattr__(self, "sparse", sparse)
        self.__post_init__()


@dataclass(frozen=True)
class DifferenceOperatorProblem:
    """Derivative orders aligned with coordinate axes, plus stencil accuracy and periodicity."""

    axes: tuple[FloatArray, ...]
    orders: tuple[int, ...]
    accuracy: int = 2
    periodic: bool = False

    def __post_init__(self) -> None:
        axes = tuple(v.array(x, "axis", ndim=1, real=True) for x in self.axes)
        if not axes or len(axes) != len(self.orders):
            raise ValueError("supply one derivative order per axis")
        v.integer(self.accuracy, "accuracy")
        v.boolean(self.periodic, "periodic")
        if self.accuracy % 2:
            raise ValueError("accuracy must be even")
        if any(
            isinstance(order, bool)
            or not isinstance(order, (int, np.integer))
            or order < 0
            for order in self.orders
        ) or not any(self.orders):
            raise ValueError(
                "orders must be nonnegative integers with at least one derivative"
            )
        for x, order in zip(axes, self.orders):
            if x.size < 2 or np.any(np.diff(x) <= 0):
                raise ValueError("operator axes must be strictly increasing")
            if order and x.size < order + self.accuracy:
                raise ValueError(
                    "grid is too short for the derivative order and accuracy"
                )
            if self.periodic and not np.allclose(
                np.diff(x), x[1] - x[0], rtol=1e-10, atol=0
            ):
                raise ValueError(
                    "periodic operators require uniform axes without a repeated endpoint"
                )
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "orders", tuple(self.orders))

    def validate_result(self, result: object) -> None:
        n = int(np.prod([x.size for x in self.axes]))
        if (
            not isinstance(result, (csr_matrix, csr_array))
            or result.shape != (n, n)
            or not np.all(np.isfinite(result.data))
        ):
            raise ValueError(
                "differentiation operator must be a finite sparse (N,N) matrix"
            )

    def __init__(
        self,
        axes: tuple[ArrayLike, ...],
        orders: tuple[int, ...],
        accuracy: int = 2,
        periodic: bool = False,
    ) -> None:
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "orders", orders)
        object.__setattr__(self, "accuracy", accuracy)
        object.__setattr__(self, "periodic", periodic)
        self.__post_init__()


@dataclass(frozen=True)
class LaplacianProblem:
    """Grid shape, uniform spacing and boundary condition for a discrete Laplacian."""

    shape: tuple[int, ...]
    spacing: float = 1.0
    boundary: str = "dirichlet"

    def __post_init__(self) -> None:
        if not self.shape:
            raise ValueError("shape cannot be empty")
        for size in self.shape:
            v.integer(size, "grid size")
        v.positive(self.spacing, "spacing")
        v.choice(self.boundary, ("dirichlet", "neumann", "periodic"), "boundary")
        object.__setattr__(self, "shape", tuple(self.shape))

    def validate_result(self, result: object) -> None:
        n = int(np.prod(self.shape))
        if (
            not isinstance(result, (csr_matrix, csr_array))
            or result.shape != (n, n)
            or not np.all(np.isfinite(result.data))
        ):
            raise ValueError("Laplacian must be a finite sparse (N,N) matrix")
