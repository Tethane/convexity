"""Public contracts shared by built-in and user-written numerical tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, TypeVar, cast

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import FloatArray, NumericArray
from .. import _validation as v


@dataclass(frozen=True)
class Diagnostics:
    """Convergence report; unavailable quantities are None, never fabricated."""

    converged: bool = True
    message: str = ""
    iterations: int | None = None
    evaluations: int | None = None
    residual_norm: float | None = None
    error_estimate: FloatArray | None = cast("FloatArray | None", None)

    def __post_init__(self) -> None:
        if not isinstance(self.converged, (bool, np.bool_)):
            raise TypeError("converged must be a boolean")
        for name in ("iterations", "evaluations"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer or None")
        if self.residual_norm is not None:
            v.positive(self.residual_norm, "residual_norm", zero=True)
        if self.error_estimate is not None:
            error = v.array(
                self.error_estimate,
                "error_estimate",
                real=True,
                finite=bool(self.converged),
            )
            if np.any(error < 0):
                raise ValueError("error_estimate must be nonnegative")
            object.__setattr__(self, "error_estimate", error)

    def __init__(
        self,
        converged: bool = True,
        message: str = "",
        iterations: int | None = None,
        evaluations: int | None = None,
        residual_norm: float | None = None,
        error_estimate: ArrayLike | None = None,
    ) -> None:
        object.__setattr__(self, "converged", converged)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "iterations", iterations)
        object.__setattr__(self, "evaluations", evaluations)
        object.__setattr__(self, "residual_norm", residual_norm)
        object.__setattr__(self, "error_estimate", error_estimate)
        self.__post_init__()


@dataclass(frozen=True, init=False)
class ArrayResult:
    """A scalar (0-D), vector, or matrix value with numerical diagnostics."""

    value: NumericArray
    diagnostics: Diagnostics = field(default_factory=Diagnostics)

    def __init__(
        self, value: ArrayLike, diagnostics: Diagnostics | None = None
    ) -> None:
        report = diagnostics if diagnostics is not None else Diagnostics()
        if not isinstance(report, Diagnostics):
            raise TypeError("diagnostics must be Diagnostics")
        object.__setattr__(
            self, "value", v.array(value, "result", finite=report.converged)
        )
        if (
            report.error_estimate is not None
            and np.shape(report.error_estimate) != self.value.shape
        ):
            raise ValueError("error_estimate must have the same shape as the result")
        object.__setattr__(self, "diagnostics", report)


class Problem(Protocol):
    """Structural numerical problem with validate_result(result); custom tools must satisfy the declared result contract."""
    def validate_result(self, result: object) -> None: ...


P = TypeVar("P", bound=Problem)
Input = TypeVar("Input", contravariant=True)
Output = TypeVar("Output", covariant=True)
R = TypeVar("R")


class NumericalTool(Protocol[Input, Output]):
    """Implement __call__, or supply an ordinary function with this signature."""

    def __call__(self, problem: Input, /) -> Output: ...


class ConvergenceError(RuntimeError):
    """A tool reported failure. The unmodified result is available as .result."""

    def __init__(self, result: object) -> None:
        self.result = result
        super().__init__(
            getattr(result, "diagnostics").message or "Numerical tool did not converge"
        )


def run(problem: P, tool: NumericalTool[P, R], *, check_convergence: bool = True) -> R:
    """Execute a compatible tool and check its output contract and convergence.

    Problem constructors validate and copy inputs before a tool is called.
    User tools own their algorithms, convergence claims, and diagnostics.
    """
    v.function(tool, "tool")
    v.boolean(check_convergence, "check_convergence")
    result = tool(problem)
    problem.validate_result(result)
    report = getattr(result, "diagnostics", None)
    if check_convergence and report is not None and not report.converged:
        raise ConvergenceError(result)
    return result


def check_array_result(
    result: object, shape: tuple[int, ...], *, real: bool = False
) -> None:
    if not isinstance(result, ArrayResult):
        raise TypeError("tool must return ArrayResult")
    if result.value.shape != shape:
        raise ValueError(f"tool returned shape {result.value.shape}; expected {shape}")
    if real and np.iscomplexobj(result.value):
        raise ValueError("tool must return real values")
