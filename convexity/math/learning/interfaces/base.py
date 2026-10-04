"""Structural extension protocols and checked prediction semantics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..._typing import _FACTORY, FloatArray
from .. import _validation as v


class PredictionAdapter(Protocol):
    """Prediction extension with input dimension, output shape and capabilities; evaluate returns aligned observation rows."""
    @property
    def input_dimension(self) -> int: ...

    @property
    def output_shape(self) -> tuple[int, ...]: ...

    @property
    def capabilities(self) -> frozenset[str]: ...

    def evaluate(
        self, operation: str, features: ArrayLike, *, offset: ArrayLike | None = None
    ) -> ArrayLike: ...


class Learner(Protocol):
    """Fit feature rows and scalar/vector responses into a FittedModel; optional training settings belong to the implementation."""
    def fit(self, features: ArrayLike, response: ArrayLike) -> "FittedModel": ...


@dataclass(frozen=True)
class Diagnostics:
    """Backend termination diagnostics. Inspect converged and message before using a fitted approximation; iterations may be unavailable."""

    backend: str = "user"
    converged: bool = True
    message: str = ""
    iterations: int | None = None

    def __post_init__(self) -> None:
        v.boolean(self.converged, "converged")
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("backend must be a nonempty string")
        if not isinstance(self.message, str):
            raise TypeError("message must be a string")
        if self.iterations is not None:
            v.integer(self.iterations, "iterations")


class ConvergenceError(RuntimeError):
    """Raised when a checked numerical method does not converge; result retains the attempted solution and diagnostics."""
    def __init__(self, result: FittedModel) -> None:
        self.result = result
        super().__init__(
            result.diagnostics.message or "Predictive fit did not converge"
        )


def checked_convergence(result: FittedModel, check: bool) -> FittedModel:
    v.boolean(check, "check_convergence")
    if check and not result.diagnostics.converged:
        raise ConvergenceError(result)
    return result


@dataclass(frozen=True)
class FittedModel:
    """A checked predictor. Capabilities state exactly which outputs exist.

    Adapters receive (n,d) and return (n,)+output_shape for means, (n,c)
    for probabilities, or (n,q) for quantiles. All leading axes are restored.
    """

    adapter: PredictionAdapter = field(repr=False)
    diagnostics: Diagnostics = field(default_factory=Diagnostics)
    nobs: int | None = None
    coefficients: FloatArray | None = cast("FloatArray | None", None)
    intercept: FloatArray | None = cast("FloatArray | None", None)
    classes: NDArray[np.generic] | None = cast("FloatArray | None", None)
    quantiles: FloatArray | None = cast("FloatArray | None", None)
    loss_history: FloatArray | None = cast("FloatArray | None", None)
    requires_offset: bool = False

    def __post_init__(self) -> None:
        v.integer(self.adapter.input_dimension, "input_dimension", 1)
        shape = self.adapter.output_shape
        if not isinstance(shape, tuple) or len(shape) > 1:
            raise ValueError("output_shape must be () or (number_of_outputs,)")
        for dimension in shape:
            v.integer(dimension, "output dimension", 1)
        caps = self.capabilities
        if not caps or not caps <= {"mean", "probability", "quantile"}:
            raise ValueError("invalid prediction capabilities")
        if "quantile" in caps and shape:
            raise ValueError("quantile models have a scalar target")
        if "probability" in caps and shape:
            raise ValueError("probability models have a scalar target")
        if not callable(self.adapter.evaluate):
            raise TypeError("adapter must implement evaluate")
        if not isinstance(self.diagnostics, Diagnostics):
            raise TypeError("diagnostics must be Diagnostics")
        if self.nobs is not None:
            v.integer(self.nobs, "nobs", 1)
        v.boolean(self.requires_offset, "requires_offset")
        if (self.classes is not None) != ("probability" in caps):
            raise ValueError("probability capability requires explicit classes")
        if self.classes is not None:
            labels = v.labels(self.classes)
            if len(labels) < 2 or len(np.unique(labels)) != len(labels):
                raise ValueError("classes must contain at least two unique labels")
            object.__setattr__(self, "classes", labels)
        if (self.quantiles is not None) != ("quantile" in caps):
            raise ValueError("quantile capability requires explicit quantile levels")
        if self.quantiles is not None:
            object.__setattr__(self, "quantiles", v.levels(self.quantiles))
        for name in ("coefficients", "intercept", "loss_history"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, v.array(value, name))
        if self.coefficients is not None:
            if (
                self.coefficients.ndim not in (1, 2)
                or self.coefficients.shape[0] != self.input_dimension
            ):
                raise ValueError("coefficients must have shape (d,) or (d,k)")
            if (
                self.intercept is None
                or self.intercept.shape != self.coefficients.shape[1:]
            ):
                raise ValueError("intercept must match coefficient output axes")
        if self.loss_history is not None and self.loss_history.ndim != 1:
            raise ValueError("loss_history must be a vector")

    @property
    def input_dimension(self):
        return self.adapter.input_dimension

    @property
    def output_shape(self):
        return self.adapter.output_shape

    @property
    def capabilities(self):
        return frozenset(self.adapter.capabilities)

    def _evaluate(
        self, operation: str, features: ArrayLike, offset: ArrayLike | None
    ) -> FloatArray:
        if operation not in self.capabilities:
            raise NotImplementedError(f"model does not support {operation} predictions")
        x = v.coordinates(features, self.input_dimension)
        leading = x.shape[:-1]
        if self.requires_offset and offset is None:
            raise ValueError("this model requires an explicit prediction offset")
        if offset is not None:
            if not getattr(self.adapter, "supports_offset", False):
                raise ValueError("this model does not support offsets")
            offset = v.array(offset, "offset", empty=True)
            if offset.shape != leading:
                raise ValueError("offset must match prediction leading axes")
            offset = offset.reshape(-1)
        tail = self.output_shape
        if operation == "probability":
            assert self.classes is not None
            tail = (len(self.classes),)
        elif operation == "quantile":
            assert self.quantiles is not None
            tail = (len(self.quantiles),)
        flat = x.reshape(-1, self.input_dimension)
        result = (
            np.empty((0,) + tail)
            if not len(flat)
            else self.adapter.evaluate(operation, flat, offset=offset)
        )
        result = v.array(result, "prediction", empty=True)
        if result.shape != (len(flat),) + tail:
            raise ValueError("prediction must preserve rows and declared output shape")
        if operation == "probability":
            if np.any((result < 0) | (result > 1)) or not np.allclose(
                result.sum(axis=-1), 1, atol=1e-10, rtol=0
            ):
                raise ValueError("class probabilities must lie in [0,1] and sum to one")
        return v.array(result.reshape(leading + tail), "prediction", empty=True)

    def predict(
        self, features: ArrayLike, *, offset: ArrayLike | None = None
    ) -> NDArray[np.generic]:
        """Mean, class label, or the declared quantile vector, in that order."""
        if "mean" in self.capabilities:
            return self.predict_mean(features, offset=offset)
        if "probability" in self.capabilities:
            return self.predict_class(features, offset=offset)
        return self.predict_quantiles(features, offset=offset)

    def predict_mean(
        self, features: ArrayLike, *, offset: ArrayLike | None = None
    ) -> FloatArray:
        return self._evaluate("mean", features, offset)

    def predict_proba(
        self, features: ArrayLike, *, offset: ArrayLike | None = None
    ) -> FloatArray:
        return self._evaluate("probability", features, offset)

    def predict_class(
        self, features: ArrayLike, *, offset: ArrayLike | None = None
    ) -> NDArray[np.generic]:
        probabilities = self.predict_proba(features, offset=offset)
        assert self.classes is not None
        result = np.array(self.classes[np.argmax(probabilities, axis=-1)], copy=True)
        result.setflags(write=False)
        return result

    def predict_quantiles(
        self, features: ArrayLike, *, offset: ArrayLike | None = None
    ) -> FloatArray:
        """Return the fitted levels in order; never sort away quantile crossings."""
        return self._evaluate("quantile", features, offset)

    def __init__(
        self,
        adapter: PredictionAdapter,
        diagnostics: Diagnostics = cast("Diagnostics", _FACTORY),
        nobs: int | None = None,
        coefficients: ArrayLike | None = None,
        intercept: ArrayLike | None = None,
        classes: ArrayLike | None = None,
        quantiles: ArrayLike | None = None,
        loss_history: ArrayLike | None = None,
        requires_offset: bool = False,
    ) -> None:
        object.__setattr__(self, "adapter", adapter)
        if diagnostics is _FACTORY:
            diagnostics = __class__.__dataclass_fields__[
                "diagnostics"
            ].default_factory()
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "nobs", nobs)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "intercept", intercept)
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "quantiles", quantiles)
        object.__setattr__(self, "loss_history", loss_history)
        object.__setattr__(self, "requires_offset", requires_offset)
        self.__post_init__()


def fit(
    features: ArrayLike, response: ArrayLike, learner: Learner, **kwargs: object
) -> FittedModel:
    """Fit a user or built-in learner and check every declared output on training rows."""
    x = v.features(features)
    # Target meaning and additional arguments belong to the learner.
    invoke = cast(Callable[..., FittedModel], learner.fit)
    result = invoke(x, response, **kwargs)
    if not isinstance(result, FittedModel):
        raise TypeError("learner must return FittedModel")
    if result.input_dimension != x.shape[1]:
        raise ValueError("fitted input_dimension must match training columns")
    for capability in result.capabilities:
        result._evaluate(capability, x, cast(ArrayLike | None, kwargs.get("offset")))
    return result
