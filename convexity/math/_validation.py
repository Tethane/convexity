"""Shared real-array ownership and basic input checks; NumPy only.

Mathematical constraints and family-specific defaults stay with each family.
Numerical mathematics has a separate array contract supporting complex values.
"""

from numbers import Integral
from collections.abc import Collection
from typing import TypeVar

import numpy as np
from numpy.typing import ArrayLike
from ._typing import FloatArray

T = TypeVar("T")


def array(
    value: object,
    name: str,
    *,
    ndim: int | None = None,
    finite: bool = True,
    empty: bool = False,
) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    result = np.array(raw, dtype=float)
    if (not empty and not result.size) or (ndim is not None and result.ndim != ndim):
        raise ValueError(f"{name} has invalid shape")
    if np.isnan(result).any() or (finite and not np.isfinite(result).all()):
        raise ValueError(f"{name} must be {'finite' if finite else 'free of NaNs'}")
    result.setflags(write=False)
    return result


def integer(value: object, name: str, minimum: int = 0) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def boolean(value: object, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be boolean")
    return bool(value)


def choice(value: T, choices: Collection[T], name: str) -> T:
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}")
    return value


def generator(rng: object) -> np.random.Generator:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an explicit numpy.random.Generator")
    return rng


def scalar(
    value: object, name: str, *, minimum: float | None = None, strict: bool = False
) -> float:
    result = float(array(value, name, ndim=0))
    if minimum is not None and (result <= minimum if strict else result < minimum):
        raise ValueError(f"{name} must be {'>' if strict else '>='} {minimum}")
    return result
