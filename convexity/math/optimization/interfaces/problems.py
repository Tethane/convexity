"""Array-first optimization problems; inequalities always use <= convention."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol, cast

import numpy as np
from numpy.typing import ArrayLike

from ... import _decision as v
from ..._typing import _FACTORY, FloatArray


@dataclass(frozen=True)
class SecondOrderCone:
    """||matrix @ x + offset||_2 <= linear @ x + bound."""

    matrix: FloatArray
    offset: FloatArray
    linear: FloatArray
    bound: float = 0.0

    def __post_init__(self) -> None:
        a = v.array(self.matrix, "cone matrix", ndim=2)
        b, c = (
            v.vector(self.offset, a.shape[0], "cone offset"),
            v.array(self.linear, "cone linear", ndim=1),
        )
        if c.shape != (a.shape[1],):
            raise ValueError("cone dimensions must align")
        for name, x in [("matrix", a), ("offset", b), ("linear", c)]:
            object.__setattr__(self, name, x)
        v.scalar(self.bound, "cone bound")

    def __init__(
        self,
        matrix: ArrayLike,
        offset: ArrayLike,
        linear: ArrayLike,
        bound: float = 0.0,
    ) -> None:
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "linear", linear)
        object.__setattr__(self, "bound", bound)
        self.__post_init__()


@dataclass(frozen=True)
class QuadraticProgram:
    """Minimize .5*x.T@quadratic@x + linear@x with affine/SOC constraints.

    Matrices have (constraint,variable) axes. Domains are continuous, integer
    or binary per variable. Bounds default to free variables, not nonnegative.
    """

    linear: FloatArray
    quadratic: FloatArray = cast("FloatArray", None)
    inequality: FloatArray | None = cast("FloatArray | None", None)
    upper: FloatArray | None = cast("FloatArray | None", None)
    equality: FloatArray | None = cast("FloatArray | None", None)
    target: FloatArray | None = cast("FloatArray | None", None)
    lower_bounds: FloatArray = cast("FloatArray", -np.inf)
    upper_bounds: FloatArray = cast("FloatArray", np.inf)
    domains: tuple[Literal["continuous", "integer", "binary"], ...] | None = None
    cones: tuple[SecondOrderCone, ...] = ()

    def __post_init__(self) -> None:
        c = v.array(self.linear, "linear", ndim=1)
        n = len(c)
        q = (
            np.zeros((n, n))
            if self.quadratic is None
            else v.psd(self.quadratic, "quadratic")
        )
        if q.shape != (n, n):
            raise ValueError("quadratic must align with variables")
        object.__setattr__(self, "linear", c)
        object.__setattr__(self, "quadratic", v.array(q, "quadratic"))
        for mat, rhs in [("inequality", "upper"), ("equality", "target")]:
            a, b = getattr(self, mat), getattr(self, rhs)
            if (a is None) != (b is None):
                raise ValueError(f"{mat} and {rhs} must be supplied together")
            if a is not None:
                a, b = v.array(a, mat, ndim=2), v.array(b, rhs, ndim=1)
                if a.shape != (len(b), n):
                    raise ValueError(f"{mat} must have (constraint,variable) shape")
                object.__setattr__(self, mat, a)
                object.__setattr__(self, rhs, b)
        lo, hi = (
            v.vector(self.lower_bounds, n, "lower bounds", finite=False),
            v.vector(self.upper_bounds, n, "upper bounds", finite=False),
        )
        if np.any(lo > hi) or np.any(lo == np.inf) or np.any(hi == -np.inf):
            raise ValueError("bounds must be ordered and admit finite points")
        object.__setattr__(self, "lower_bounds", lo)
        object.__setattr__(self, "upper_bounds", hi)
        domains = tuple(["continuous"] * n if self.domains is None else self.domains)
        if len(domains) != n or any(
            d not in ("continuous", "integer", "binary") for d in domains
        ):
            raise ValueError(
                "one continuous/integer/binary domain per variable is required"
            )
        object.__setattr__(self, "domains", domains)
        cones = tuple(self.cones)
        if any(
            not isinstance(k, SecondOrderCone) or k.linear.shape != c.shape
            for k in cones
        ):
            raise ValueError("cones must align with variable dimension")
        object.__setattr__(self, "cones", cones)

    def __init__(
        self,
        linear: ArrayLike,
        quadratic: ArrayLike | None = None,
        inequality: ArrayLike | None = None,
        upper: ArrayLike | None = None,
        equality: ArrayLike | None = None,
        target: ArrayLike | None = None,
        lower_bounds: ArrayLike = -float("inf"),
        upper_bounds: ArrayLike = float("inf"),
        domains: Sequence[Literal["continuous", "integer", "binary"]] | None = None,
        cones: tuple[SecondOrderCone, ...] = (),
    ) -> None:
        object.__setattr__(self, "linear", linear)
        object.__setattr__(self, "quadratic", quadratic)
        object.__setattr__(self, "inequality", inequality)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "equality", equality)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "lower_bounds", lower_bounds)
        object.__setattr__(self, "upper_bounds", upper_bounds)
        object.__setattr__(self, "domains", domains)
        object.__setattr__(self, "cones", cones)
        self.__post_init__()


@dataclass(frozen=True)
class NonlinearProblem:
    """Minimize objective(x), with inequalities(x)<=0 and equalities(x)==0."""

    objective: Callable[[FloatArray], float]
    initial: FloatArray
    lower_bounds: FloatArray = cast("FloatArray", -np.inf)
    upper_bounds: FloatArray = cast("FloatArray", np.inf)
    inequalities: Callable[[FloatArray], ArrayLike] | None = None
    equalities: Callable[[FloatArray], ArrayLike] | None = None
    gradient: Callable[[FloatArray], ArrayLike] | None = None

    def __post_init__(self) -> None:
        x = v.array(self.initial, "initial", ndim=1)
        if not callable(self.objective):
            raise TypeError("objective must be callable")
        for name in ("inequalities", "equalities", "gradient"):
            if getattr(self, name) is not None and not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")
        lo, hi = (
            v.vector(self.lower_bounds, len(x), "lower bounds", finite=False),
            v.vector(self.upper_bounds, len(x), "upper bounds", finite=False),
        )
        if np.any(lo > hi) or np.any(x < lo) or np.any(x > hi):
            raise ValueError("initial point must satisfy ordered bounds")
        for name, y in [("initial", x), ("lower_bounds", lo), ("upper_bounds", hi)]:
            object.__setattr__(self, name, y)

    def __init__(
        self,
        objective: Callable[[FloatArray], float],
        initial: ArrayLike,
        lower_bounds: ArrayLike = -float("inf"),
        upper_bounds: ArrayLike = float("inf"),
        inequalities: Callable[[FloatArray], ArrayLike] | None = None,
        equalities: Callable[[FloatArray], ArrayLike] | None = None,
        gradient: Callable[[FloatArray], ArrayLike] | None = None,
    ) -> None:
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "lower_bounds", lower_bounds)
        object.__setattr__(self, "upper_bounds", upper_bounds)
        object.__setattr__(self, "inequalities", inequalities)
        object.__setattr__(self, "equalities", equalities)
        object.__setattr__(self, "gradient", gradient)
        self.__post_init__()


@dataclass(frozen=True)
class SolveResult:
    """Optimization solution and termination diagnostics; success is not a global-optimality certificate for nonconvex problems."""

    solution: FloatArray | None
    objective: float | None
    status: str
    solver: str
    max_violation: float | None = None
    duals: Mapping[str, FloatArray] = field(default_factory=dict)
    gap: float | None = None
    iterations: int | None = None
    message: str = ""

    def __post_init__(self) -> None:
        if self.solution is not None:
            object.__setattr__(
                self, "solution", v.array(self.solution, "solution", ndim=1)
            )
        if self.objective is not None:
            v.scalar(self.objective, "objective")
        for name in ("max_violation", "gap"):
            if getattr(self, name) is not None:
                v.scalar(getattr(self, name), name, minimum=0)
        object.__setattr__(
            self,
            "duals",
            MappingProxyType(
                {k: v.array(x, f"dual {k}") for k, x in self.duals.items()}
            ),
        )

    @property
    def success(self):
        return self.status in ("optimal", "local_optimum") and self.solution is not None

    def __init__(
        self,
        solution: ArrayLike | None,
        objective: float | None,
        status: str,
        solver: str,
        max_violation: float | None = None,
        duals: Mapping[str, ArrayLike] = cast("Mapping[str, ArrayLike]", _FACTORY),
        gap: float | None = None,
        iterations: int | None = None,
        message: str = "",
    ) -> None:
        object.__setattr__(self, "solution", solution)
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "solver", solver)
        object.__setattr__(self, "max_violation", max_violation)
        if duals is _FACTORY:
            duals = __class__.__dataclass_fields__["duals"].default_factory()
        object.__setattr__(self, "duals", duals)
        object.__setattr__(self, "gap", gap)
        object.__setattr__(self, "iterations", iterations)
        object.__setattr__(self, "message", message)
        self.__post_init__()


class Solver[ProblemT = object](Protocol):
    """Solve a built-in or custom optimization problem, retaining its type and returning SolveResult."""

    def solve(self, problem: ProblemT) -> SolveResult: ...


class OptimizationError(RuntimeError):
    """Raised when optimization cannot supply an acceptable solution; inspect retained status and diagnostics."""

    def __init__(self, result: SolveResult) -> None:
        self.result = result
        super().__init__(f"{result.solver}: {result.status}: {result.message}")
