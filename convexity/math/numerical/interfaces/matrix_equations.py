"""Standard continuous/discrete Lyapunov and algebraic Riccati equations."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from .. import _validation as v
from .base import check_array_result

TimeDomain = Literal["continuous", "discrete"]


@dataclass(frozen=True)
class LyapunovEquation:
    """Continuous: AX + XA^H + Q = 0. Discrete: AXA^H - X + Q = 0."""

    a: ArrayLike
    q: ArrayLike
    time: TimeDomain = "continuous"

    def __post_init__(self) -> None:
        v.choice(self.time, ("continuous", "discrete"), "time")
        a, q = v.square(self.a, "a"), v.square(self.q, "q")
        if a.shape != q.shape:
            raise ValueError("a and q must have identical shapes")
        v.hermitian(q, "q")
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "q", q)

    def validate_result(self, result: object) -> None:
        check_array_result(result, np.shape(self.a))


@dataclass(frozen=True)
class RiccatiEquation:
    """CARE: A^H X + XA - XBR^-1 B^H X + Q = 0.

    DARE: A^H XA - X - A^H XB(R+B^H XB)^-1 B^H XA + Q = 0.
    Q and R are Hermitian. Existence/stabilizability is checked by the solver.
    """

    a: ArrayLike
    b: ArrayLike
    q: ArrayLike
    r: ArrayLike
    time: TimeDomain = "continuous"

    def __post_init__(self) -> None:
        v.choice(self.time, ("continuous", "discrete"), "time")
        a, b = v.square(self.a, "a"), v.array(self.b, "b", ndim=2)
        q, r = v.square(self.q, "q"), v.square(self.r, "r")
        if (
            q.shape != a.shape
            or b.shape[0] != a.shape[0]
            or r.shape != (b.shape[1], b.shape[1])
        ):
            raise ValueError("expected a/q: (n,n), b: (n,m), r: (m,m)")
        v.hermitian(q, "q")
        v.hermitian(r, "r")
        for name, value in (("a", a), ("b", b), ("q", q), ("r", r)):
            object.__setattr__(self, name, value)

    def validate_result(self, result: object) -> None:
        check_array_result(result, np.shape(self.a))
