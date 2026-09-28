"""Real ODE systems, event contracts, and checked continuous solutions."""
from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np
from .. import _validation as v
from .base import Diagnostics


@dataclass(frozen=True)
class ODEEvent:
    function: Callable
    terminal: bool = False
    direction: int = 0

    def __post_init__(self):
        v.function(self.function, "event")
        v.boolean(self.terminal, "terminal")
        if self.direction not in (-1, 0, 1):
            raise ValueError("event direction must be -1, 0, or 1")


@dataclass(frozen=True)
class InitialValueProblem:
    function: Callable
    y0: Any
    t_span: tuple[float, float]
    t_eval: Any = None
    events: tuple[ODEEvent, ...] = ()
    jacobian: Callable | None = None

    def __post_init__(self):
        v.function(self.function)
        y = v.array(self.y0, "y0", ndim=1, real=True)
        span = v.array(self.t_span, "t_span", ndim=1, real=True)
        if span.shape != (2,) or span[0] == span[1]:
            raise ValueError("t_span must contain two distinct finite endpoints")
        if self.t_eval is not None:
            t = v.array(self.t_eval, "t_eval", ndim=1, real=True)
            if np.any(np.diff(t)*np.sign(span[1]-span[0]) <= 0) or np.any((t < min(span)) | (t > max(span))):
                raise ValueError("t_eval must be strictly ordered in the integration direction and within t_span")
            object.__setattr__(self, "t_eval", t)
        if any(not isinstance(event, ODEEvent) for event in self.events):
            raise TypeError("events must contain ODEEvent objects")
        if self.jacobian is not None:
            v.function(self.jacobian, "jacobian")
        object.__setattr__(self, "y0", y)
        object.__setattr__(self, "t_span", (float(span[0]), float(span[1])))
        object.__setattr__(self, "events", tuple(self.events))

    def validate_result(self, result):
        if not isinstance(result, ODEResult):
            raise TypeError("tool must return ODEResult")
        if result.y.shape != (np.size(self.y0), result.t.size) or len(result.events) != len(self.events):
            raise ValueError("invalid ODE result shapes or event count")
        lo, hi = sorted(self.t_span)
        if np.any((result.t < lo) | (result.t > hi)) or np.any(np.diff(result.t)*np.sign(self.t_span[1]-self.t_span[0]) <= 0):
            raise ValueError("ODE times must lie in the requested span and direction")
        if result.solution.state_size != np.size(self.y0):
            raise ValueError("dense solution has the wrong state size")
        domain = result.solution.domain
        if domain[0] < lo or domain[1] > hi or np.any((result.t < domain[0]) | (result.t > domain[1])):
            raise ValueError("dense solution domain does not match the solved interval")
        if result.diagnostics.converged and not result.terminated_by_event and domain != (lo, hi):
            raise ValueError("successful integration must reach the endpoint or report a terminal event")


@dataclass(frozen=True)
class DenseSolution:
    _evaluate: Callable = field(repr=False)
    domain: tuple[float, float]
    state_size: int

    def __post_init__(self):
        v.function(self._evaluate)
        v.integer(self.state_size, "state_size")
        domain = v.array(self.domain, "domain", ndim=1, real=True)
        if domain.shape != (2,) or domain[0] > domain[1]:
            raise ValueError("dense solution domain must be ordered")
        object.__setattr__(self, "domain", (float(domain[0]), float(domain[1])))

    def __call__(self, t):
        points = v.array(t, "times", real=True)
        if points.ndim > 1 or np.any((points < self.domain[0]) | (points > self.domain[1])):
            raise ValueError("dense solution queries must be scalar/vector and inside the solved domain")
        result = v.array(self._evaluate(points), "solution", real=True)
        if result.shape != (self.state_size,) + points.shape:
            raise ValueError("dense solution output has the wrong shape")
        return result


@dataclass(frozen=True)
class EventResult:
    t: Any
    y: Any

    def __post_init__(self):
        t, y = v.array(self.t, "event times", ndim=1, real=True, allow_empty=True), v.array(self.y, "event states", ndim=2, real=True, allow_empty=True)
        if y.shape[0] != t.size:
            raise ValueError("event states must have shape (events, states)")
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "y", y)


@dataclass(frozen=True)
class ODEResult:
    t: Any
    y: Any
    solution: DenseSolution
    diagnostics: Diagnostics = Diagnostics()
    events: tuple[EventResult, ...] = ()
    terminated_by_event: bool = False

    def __post_init__(self):
        if not isinstance(self.diagnostics, Diagnostics) or not isinstance(self.solution, DenseSolution):
            raise TypeError("expected Diagnostics and DenseSolution")
        t = v.array(self.t, "times", ndim=1, real=True, allow_empty=True)
        y = v.array(self.y, "states", ndim=2, real=True, finite=self.diagnostics.converged, allow_empty=True)
        if y.shape != (self.solution.state_size, t.size):
            raise ValueError("states must have shape (state_size, times)")
        if any(not isinstance(e, EventResult) or e.y.shape[1] != y.shape[0] for e in self.events):
            raise ValueError("invalid event states")
        v.boolean(self.terminated_by_event, "terminated_by_event")
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "events", tuple(self.events))


@dataclass(frozen=True)
class BoundaryValueProblem:
    """y'=f(x,y), bc(y(a),y(b))=0. Callback states have shape (n, mesh_size)."""
    function: Callable
    boundary: Callable
    x: Any
    y: Any

    def __post_init__(self):
        v.function(self.function)
        v.function(self.boundary, "boundary")
        x, y = v.array(self.x, "x", ndim=1, real=True), v.array(self.y, "y", ndim=2, real=True)
        if x.size < 2 or np.any(np.diff(x) <= 0) or y.shape[1] != x.size:
            raise ValueError("x must increase strictly; initial y must have shape (n, len(x))")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)

    def validate_result(self, result):
        if not isinstance(result, BoundaryValueResult):
            raise TypeError("tool must return BoundaryValueResult")
        if result.y.shape != (np.shape(self.y)[0], result.x.size) or result.x[0] != np.asarray(self.x)[0] or result.x[-1] != np.asarray(self.x)[-1]:
            raise ValueError("invalid BVP solution shape/domain")


@dataclass(frozen=True)
class BoundaryValueResult:
    x: Any
    y: Any
    solution: DenseSolution
    rms_residuals: Any
    diagnostics: Diagnostics

    def __post_init__(self):
        if not isinstance(self.diagnostics, Diagnostics) or not isinstance(self.solution, DenseSolution):
            raise TypeError("expected Diagnostics and DenseSolution")
        for name, ndim in (("x", 1), ("y", 2), ("rms_residuals", 1)):
            object.__setattr__(self, name, v.array(getattr(self, name), name, ndim=ndim, real=True, finite=self.diagnostics.converged))
        if np.any(np.diff(self.x) <= 0) or self.rms_residuals.shape != (self.x.size-1,) or np.any(self.rms_residuals < 0):
            raise ValueError("invalid BVP mesh or residuals")
        if self.y.shape != (self.solution.state_size, self.x.size) or self.solution.domain != (self.x[0], self.x[-1]):
            raise ValueError("BVP solution shape/domain mismatch")
