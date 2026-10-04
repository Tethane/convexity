"""Exact laws and sample-only kernels have different declared capabilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import (
    FloatArray,
    SampleSize,
)

if TYPE_CHECKING:
    from ...distributions import Distribution
from .. import _validation as v


@dataclass(frozen=True)
class Transition:
    """A conditional kernel for one fixed initial state and elapsed interval.

    sampler(size, rng) returns size + event_shape. A Distribution is optional:
    numerical kernels and mixed jump laws need not pretend to have a density.
    mean/covariance are analytic moments when supplied, never Monte Carlo fits.
    """

    event_shape: tuple[int, ...]
    method: str
    backend: str
    exact: bool
    _sampler: Callable[[tuple[int, ...], np.random.Generator], ArrayLike] = field(
        repr=False
    )
    _law: Distribution | None = field(default=None, repr=False)
    _mean: FloatArray | None = field(default=None, repr=False)
    _covariance: FloatArray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        v.shape(self.event_shape)
        v.boolean(self.exact, "exact")
        if not callable(self._sampler):
            raise TypeError("sampler must be callable")
        for name in ("method", "backend"):
            v.label(getattr(self, name), name)
        if self._law is not None:
            from ...distributions import Distribution

            if (
                not isinstance(self._law, Distribution)
                or self._law.event_shape != self.event_shape
            ):
                raise ValueError(
                    "law must be a Distribution with the kernel event_shape"
                )
        for name, shape in (
            ("_mean", self.event_shape),
            ("_covariance", self.event_shape * 2),
        ):
            if getattr(self, name) is not None:
                object.__setattr__(
                    self, name, v.shaped(getattr(self, name), shape, name)
                )
        if self._covariance is not None:
            object.__setattr__(
                self, "_covariance", v.covariance(self._covariance, self.event_shape)
            )
        if self._law is not None:
            for name in ("mean", "covariance"):
                supplied = getattr(self, "_" + name)
                if supplied is not None and name in self._law.capabilities:
                    expected = getattr(self._law, name)()
                    if not np.allclose(supplied, expected, rtol=1e-10, atol=0):
                        raise ValueError(
                            f"explicit {name} must agree with the Distribution"
                        )

    @classmethod
    def identity(cls, state: ArrayLike):
        """Zero-duration kernel; sampling consumes no random numbers."""
        from .._backends.laws import atom

        state = v.array(state, "state")
        return cls(
            state.shape,
            "exact",
            "numpy",
            True,
            lambda size, rng: np.broadcast_to(state, size + state.shape),
            atom(state),
        )

    @classmethod
    def from_distribution(
        cls, law, *, method: str = "exact", backend: str = "scipy", exact: bool = True
    ):
        return cls(
            law.event_shape,
            method,
            backend,
            exact,
            lambda size, rng: law.sample(size, rng=rng),
            law,
        )

    @property
    def capabilities(self):
        result = {"sample"}
        if self._law is not None:
            result.add("distribution")
        for name in ("mean", "covariance"):
            if getattr(self, "_" + name) is not None or (
                self._law is not None and name in self._law.capabilities
            ):
                result.add(name)
        return frozenset(result)

    def sample(self, size: SampleSize = (), *, rng: np.random.Generator):
        size = v.sample_shape(size)
        v.generator(rng)
        if 0 in size:
            return v.shaped(
                np.empty(size + self.event_shape), size + self.event_shape, "samples"
            )
        return v.shaped(self._sampler(size, rng), size + self.event_shape, "samples")

    def distribution(self):
        if self._law is None:
            raise NotImplementedError(
                "this transition supplies samples, not a Distribution"
            )
        return self._law

    def mean(self):
        if self._mean is not None:
            return self._mean
        return self.distribution().mean()

    def covariance(self):
        if self._covariance is not None:
            return self._covariance
        return self.distribution().covariance()

    def __init__(
        self,
        event_shape: tuple[int, ...],
        method: str,
        backend: str,
        exact: bool,
        _sampler: Callable[[tuple[int, ...], np.random.Generator], ArrayLike],
        _law: Distribution | None = None,
        _mean: ArrayLike | None = None,
        _covariance: ArrayLike | None = None,
    ) -> None:
        object.__setattr__(self, "event_shape", event_shape)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "exact", exact)
        object.__setattr__(self, "_sampler", _sampler)
        object.__setattr__(self, "_law", _law)
        object.__setattr__(self, "_mean", _mean)
        object.__setattr__(self, "_covariance", _covariance)
        self.__post_init__()


@dataclass(frozen=True)
class ProcessPaths:
    """Joint draws (draw,time,*event_shape), including the supplied initial state."""

    times: FloatArray
    values: FloatArray
    method: str
    backend: str
    exact: bool
    methods: tuple[str, ...] = ()
    backends: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        times = v.times(self.times)
        values = v.array(self.values, "values")
        if values.ndim not in (2, 3) or values.shape[1] != len(times):
            raise ValueError(
                "values must have shape (draw,time) or (draw,time,dimension)"
            )
        v.boolean(self.exact, "exact")
        for singular, plural in (("method", "methods"), ("backend", "backends")):
            summary = v.label(getattr(self, singular), singular)
            raw = getattr(self, plural)
            if isinstance(raw, str):
                raise ValueError(f"{plural} must be a sequence of labels")
            labels = tuple(dict.fromkeys(v.label(x, singular) for x in raw)) or (
                summary,
            )
            if summary != (labels[0] if len(labels) == 1 else "mixed"):
                raise ValueError(f"{singular} must summarize {plural}")
            object.__setattr__(self, plural, labels)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "values", values)

    def marginal(self, index: int):
        """An explicitly empirical marginal, even when the transition is exact."""
        from ...distributions import empirical

        index = v.integer(index, "index")
        if index >= len(self.times):
            raise ValueError("time index out of range")
        return empirical(self.values[:, index])

    def __init__(
        self,
        times: ArrayLike,
        values: ArrayLike,
        method: str,
        backend: str,
        exact: bool,
        methods: tuple[str, ...] = (),
        backends: tuple[str, ...] = (),
    ) -> None:
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "exact", exact)
        object.__setattr__(self, "methods", methods)
        object.__setattr__(self, "backends", backends)
        self.__post_init__()
