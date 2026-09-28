"""A checked law facade and a small, structural adapter contract."""

from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np

from .. import _validation as v

Measure = Literal["continuous", "discrete"]


class DistributionAdapter(Protocol):
    """Implement only advertised operations; no inheritance or registry required.

    Events are scalar () or vector (d,). There are no parameter batch axes.
    evaluate receives already checked inputs; see Distribution for output shapes.
    Adapters must retain immutable parameters and not mutate callback arguments.
    """

    @property
    def event_shape(self) -> tuple[int, ...]: ...

    @property
    def measure(self) -> Measure: ...

    @property
    def capabilities(self) -> frozenset[str]: ...

    def evaluate(self, operation: str, *args: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class Distribution:
    """One NumPy probability law. Unsupported operations raise NotImplementedError.

    density means PDF for continuous laws and PMF for discrete laws. Vector CDF
    means P(X_i <= x_i for every i), not a vector of marginal CDFs. Outputs are
    read-only arrays, including 0-D arrays for scalar outputs. NaN/Inf moments
    retain the backend's undefined/divergent moment semantics.
    """

    _adapter: DistributionAdapter

    def __post_init__(self):
        shape = self._adapter.event_shape
        if not isinstance(shape, tuple) or len(shape) > 1:
            raise ValueError("event_shape must be () or (dimension,)")
        if shape:
            v.integer(shape[0], "event dimension", minimum=1)
        if self.measure not in ("continuous", "discrete"):
            raise ValueError("measure must be continuous or discrete")
        if not {"log_density", "sample"} <= self.capabilities:
            raise ValueError("a law must support log_density and sample")
        if not callable(self._adapter.evaluate):
            raise TypeError("adapter.evaluate must be callable")

    @property
    def event_shape(self):
        return self._adapter.event_shape

    @property
    def measure(self):
        return self._adapter.measure

    @property
    def capabilities(self):
        capabilities = frozenset(self._adapter.capabilities) | {"density"}
        if "variance" in capabilities:
            capabilities |= {"std"}
            if not self.event_shape:
                capabilities |= {"covariance"}
        return capabilities

    def _call(self, operation, *args, **kwargs):
        if operation not in self.capabilities:
            raise NotImplementedError(f"this law does not support {operation}")
        return self._adapter.evaluate(operation, *args, **kwargs)

    def _points(self, value):
        x = v.array(value, "evaluation points", finite=False, empty=True)
        if self.event_shape and (x.ndim == 0 or x.shape[-1:] != self.event_shape):
            raise ValueError(f"points must have trailing event shape {self.event_shape}")
        return x, x.shape[:-1] if self.event_shape else x.shape

    @staticmethod
    def _result(value, shape=None, *, moments=False):
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            raise ValueError("adapter must return real numeric values")
        result = np.array(raw, dtype=float)
        if shape is not None and result.shape != shape:
            raise ValueError(f"adapter returned shape {result.shape}; expected {shape}")
        if not moments and np.any(np.isnan(result)):
            raise ValueError("adapter returned NaN")
        result.setflags(write=False)
        return result

    def log_density(self, x):
        points, shape = self._points(x)
        return self._result(self._call("log_density", points), shape)

    def density(self, x):
        return self._result(np.exp(self.log_density(x)))

    def cdf(self, x, *, rng=None):
        points, shape = self._points(x)
        if rng is not None:
            v.generator(rng)
        result = self._result(self._call("cdf", points, rng=rng), shape)
        if np.any((result < 0) | (result > 1)):
            raise ValueError("adapter CDF must lie in [0, 1]")
        return result

    def survival(self, x):
        """P(X > x) for scalar laws; SciPy laws use the stable backend SF."""
        points, shape = self._points(x)
        result = self._result(self._call("survival", points), shape)
        if np.any((result < 0) | (result > 1)):
            raise ValueError("adapter survival probability must lie in [0, 1]")
        return result

    def quantile(self, probability):
        p = v.array(probability, "probability", empty=True)
        if np.any((p < 0) | (p > 1)):
            raise ValueError("probability must lie in [0, 1]")
        return self._result(self._call("quantile", p), p.shape)

    def sample(self, size=(), *, rng):
        """Return size + event_shape draws using only the supplied generator."""
        shape = v.sample_shape(size)
        v.generator(rng)
        result = self._result(self._call("sample", shape, rng=rng), shape + self.event_shape)
        if not np.all(np.isfinite(result)):
            raise ValueError("adapter returned nonfinite samples")
        return result

    def mean(self):
        return self._result(self._call("mean"), self.event_shape, moments=True)

    def variance(self):
        return self._result(self._call("variance"), self.event_shape, moments=True)

    def std(self):
        return self._result(np.sqrt(self.variance()), moments=True)

    def covariance(self):
        """Scalar variance or vector (d,d) covariance, never an estimated matrix."""
        if not self.event_shape:
            return self.variance()
        return self._result(self._call("covariance"), self.event_shape * 2, moments=True)

    def moment(self, order):
        """Raw E[X**order], componentwise for vector laws where supported."""
        v.integer(order, "order")
        return self._result(self._call("moment", order), self.event_shape, moments=True)

    def expect(self, function, **options):
        """Expectation of a one-event callback; no implicit sampling fallback.

        Scalar SciPy laws integrate a scalar callback (options go to SciPy
        expect). Empirical laws sum scalar/vector/matrix callback outputs exactly.
        The caller must establish integrability; quadrature cannot prove it.
        """
        if not callable(function):
            raise TypeError("function must be callable")
        return self._result(self._call("expect", function, **options), moments=True)

    def support(self):
        """Scalar support endpoints; quantile(0/1) uses these endpoints."""
        result = self._result(self._call("support"), (2,))
        if result[0] > result[1]:
            raise ValueError("invalid support endpoints")
        return tuple(float(x) for x in result)

    def marginal(self, indices):
        if not self.event_shape:
            raise ValueError("marginal requires a vector law")
        keep = v.indices(indices, self.event_shape[0])
        result = self._call("marginal", keep)
        if not isinstance(result, Distribution) or result.event_shape != (len(keep),):
            raise ValueError("marginal must return a vector law in the requested coordinate order")
        return result

    def condition(self, indices, values):
        """Law of the remaining coordinates given X[indices]=values.

        Remaining coordinates retain original order, including a one-coordinate
        vector. Continuous Gaussian/t laws use regular conditional densities;
        empirical laws require positive mass at exact observed values.
        """
        if not self.event_shape:
            raise ValueError("coordinate conditioning requires a vector law")
        observed = v.indices(indices, self.event_shape[0], proper=True)
        values = v.array(values, "observed values", ndim=1)
        if values.shape != observed.shape:
            raise ValueError("values must align with observed indices")
        result = self._call("condition", observed, values)
        expected = (self.event_shape[0] - len(observed),)
        if not isinstance(result, Distribution) or result.event_shape != expected:
            raise ValueError("condition must return the law of the remaining coordinates")
        return result

    def probability(self, event):
        """Exact predicate probability for finite empirical laws."""
        if not callable(event):
            raise TypeError("event must be callable")
        result = self._result(self._call("probability", event), ())
        if not 0 <= result <= 1:
            raise ValueError("event probability must lie in [0, 1]")
        return result

    def given(self, event):
        """Restrict a finite empirical law to a positive-probability predicate."""
        if not callable(event):
            raise TypeError("event must be callable")
        result = self._call("given", event)
        if not isinstance(result, Distribution) or result.event_shape != self.event_shape:
            raise ValueError("given must preserve the event shape")
        return result

    def map(self, function):
        """Exact finite pushforward; arbitrary scalar/vector maps of empirical laws."""
        if not callable(function):
            raise TypeError("function must be callable")
        result = self._call("map", function)
        if not isinstance(result, Distribution):
            raise TypeError("map must return a Distribution")
        return result
