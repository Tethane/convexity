"""Owned chain/draw arrays; finite posterior laws retain joint dependence."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arviz import InferenceData
    from pymc import Model
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, cast

import numpy as np
from numpy.typing import ArrayLike

from ..._typing import _FACTORY, FloatArray
from .. import _validation as v


@dataclass(frozen=True)
class InferenceDiagnostics:
    """Inference status and available sampling diagnostics; absence of a metric is not evidence of convergence."""

    method: str
    backend: str = "user"
    rhat: Mapping[str, FloatArray] | None = cast(
        "Mapping[str, FloatArray] | None", None
    )
    ess_bulk: Mapping[str, FloatArray] | None = cast(
        "Mapping[str, FloatArray] | None", None
    )
    ess_tail: Mapping[str, FloatArray] | None = cast(
        "Mapping[str, FloatArray] | None", None
    )
    divergences: int | None = None
    loss: FloatArray | None = cast("FloatArray | None", None)
    note: str = ""

    def __post_init__(self) -> None:
        for name in ("method", "backend"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty string")
        for name in ("rhat", "ess_bulk", "ess_tail"):
            value = getattr(self, name)
            if value is not None:
                # NaN is meaningful for constant variables / insufficient chains.
                copied = {}
                for key, raw in value.items():
                    a = np.array(raw, dtype=float, copy=True)
                    a.setflags(write=False)
                    copied[key] = a
                object.__setattr__(self, name, MappingProxyType(copied))
        if self.divergences is not None:
            v.integer(self.divergences, "divergences")
        if self.loss is not None:
            object.__setattr__(self, "loss", v.array(self.loss, "loss", ndim=1))

    def __init__(
        self,
        method: str,
        backend: str = "user",
        rhat: Mapping[str, ArrayLike] | None = None,
        ess_bulk: Mapping[str, ArrayLike] | None = None,
        ess_tail: Mapping[str, ArrayLike] | None = None,
        divergences: int | None = None,
        loss: ArrayLike | None = None,
        note: str = "",
    ) -> None:
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "rhat", rhat)
        object.__setattr__(self, "ess_bulk", ess_bulk)
        object.__setattr__(self, "ess_tail", ess_tail)
        object.__setattr__(self, "divergences", divergences)
        object.__setattr__(self, "loss", loss)
        object.__setattr__(self, "note", note)
        self.__post_init__()


@dataclass(frozen=True)
class PosteriorSamples:
    """samples[name] has shape (chain,draw,*event); warmup is excluded.

    MCMC draws remain correlated. VI uses one draw axis, not a Markov chain.
    distribution() is a finite empirical law, not an estimated continuous PDF.
    """

    samples: Mapping[str, FloatArray]
    diagnostics: InferenceDiagnostics = field(
        default_factory=lambda: InferenceDiagnostics("user")
    )
    _model: Model | None = field(default=None, repr=False, compare=False)
    _trace: InferenceData | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.diagnostics, InferenceDiagnostics):
            raise TypeError("diagnostics must be InferenceDiagnostics")
        if not self.samples:
            raise ValueError("samples must contain at least one named variable")
        copied = {}
        shape = None
        for name, value in self.samples.items():
            if not isinstance(name, str) or not name:
                raise ValueError("sample names must be nonempty strings")
            a = v.array(value, name)
            if a.ndim < 2 or (shape is not None and a.shape[:2] != shape):
                raise ValueError(
                    "samples must share nonempty (chain,draw) leading axes"
                )
            shape = a.shape[:2]
            copied[name] = a
        object.__setattr__(self, "samples", MappingProxyType(copied))

    @property
    def chains(self):
        return next(iter(self.samples.values())).shape[0]

    @property
    def draws(self):
        return next(iter(self.samples.values())).shape[1]

    def mean(self, name) -> FloatArray:
        return v.array(self.samples[name].mean(axis=(0, 1)), "posterior mean")

    def interval(self, name, level: float = 0.95) -> FloatArray:
        tail = (1 - v.probability(level)) / 2
        return v.array(
            np.quantile(self.samples[name], [tail, 1 - tail], axis=(0, 1)),
            "credible interval",
        )

    def distribution(self, names: Sequence[str] | None = None):
        from ...distributions import empirical

        selected = tuple(self.samples) if names is None else tuple(names)
        if (
            not selected
            or len(set(selected)) != len(selected)
            or any(n not in self.samples for n in selected)
        ):
            raise ValueError("names must select distinct posterior variables")
        flat = [self.samples[n].reshape(self.chains * self.draws, -1) for n in selected]
        return empirical(np.concatenate(flat, axis=1))

    def posterior_predictive(
        self,
        *,
        rng: np.random.Generator,
        names: Sequence[str] | None = None,
        model=None,
    ):
        """PyMC forward draws, optionally in a compatible fresh prediction model."""
        from .._backends.pymc import predict

        return predict(self, names=names, model=model, rng=rng)

    def __init__(
        self,
        samples: Mapping[str, ArrayLike],
        diagnostics: InferenceDiagnostics = cast("InferenceDiagnostics", _FACTORY),
        _model: Model | None = None,
        _trace: InferenceData | None = None,
    ) -> None:
        object.__setattr__(self, "samples", samples)
        if diagnostics is _FACTORY:
            diagnostics = __class__.__dataclass_fields__[
                "diagnostics"
            ].default_factory()
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "_model", _model)
        object.__setattr__(self, "_trace", _trace)
        self.__post_init__()
