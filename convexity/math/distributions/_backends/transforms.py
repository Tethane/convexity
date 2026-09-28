"""Change-of-variables algebra for continuous scalar laws."""

from dataclasses import dataclass
from math import comb
from typing import Any

import numpy as np
from scipy import integrate

from .. import _validation as v
from ..interfaces import Distribution, Measure, MonotoneTransform


@dataclass(frozen=True)
class TransformedAdapter:
    base: Distribution
    transform: MonotoneTransform
    measure: Measure = "continuous"
    event_shape: tuple[int, ...] = ()
    affine_parameters: tuple[float, float] | None = None

    @property
    def capabilities(self):
        common = {"log_density", "sample"}
        if {"cdf", "survival"} <= self.base.capabilities:
            common |= {"cdf", "survival"}
        common |= self.base.capabilities & {"quantile", "support"}
        if "support" in self.base.capabilities:
            common.add("expect")
        if self.affine_parameters is not None:
            common |= self.base.capabilities & {"mean", "variance", "moment"}
        return frozenset(common)

    @staticmethod
    def _apply(function, x, *, finite=False):
        result = v.array(function(x), "transform callback", finite=finite, empty=True)
        if result.shape != np.shape(x):
            raise ValueError("transform callback must preserve input shape")
        return result

    def evaluate(self, operation, *args, **kwargs) -> Any:
        t = self.transform
        if operation == "sample":
            return self._apply(t.forward, self.base.sample(args[0], **kwargs), finite=True)
        if operation == "support":
            lower, upper = self.base.support()
            result = []
            for x in (lower, upper):
                if np.isneginf(x):
                    result.append(t.image[0 if t.increasing else 1])
                elif np.isposinf(x):
                    result.append(t.image[1 if t.increasing else 0])
                else:
                    result.append(float(self._apply(t.forward, np.asarray(x))))
            return sorted(result)
        if operation == "quantile":
            p = args[0] if t.increasing else 1 - args[0]
            x = self.base.quantile(p)
            result = np.empty(p.shape)
            finite = np.isfinite(x)
            result[finite] = self._apply(t.forward, x[finite])
            result[np.isneginf(x)] = t.image[0 if t.increasing else 1]
            result[np.isposinf(x)] = t.image[1 if t.increasing else 0]
            return result
        if operation in {"log_density", "cdf", "survival"}:
            y = args[0]
            lo, hi = t.image
            inside = (y > lo) & (y < hi)
            x = self._apply(t.inverse, y[inside], finite=True)
            if operation == "log_density":
                result = np.full(y.shape, -np.inf)
                result[inside] = self.base.log_density(x) - self._apply(t.log_abs_derivative, x, finite=True)
            else:
                result = np.zeros(y.shape)
                if operation == "cdf":
                    result[y >= hi] = 1.
                    method = self.base.cdf if t.increasing else self.base.survival
                else:
                    result[y <= lo] = 1.
                    method = self.base.survival if t.increasing else self.base.cdf
                result[inside] = method(x)
            return result
        if operation == "expect":
            if {"lb", "ub", "conditional", "full_output", "complex_func"} & kwargs.keys():
                raise ValueError("expect integrates a real callback over the full transformed law")
            # Integrate in the image coordinate. Integrating exp(x)*normal.pdf(x)
            # through a base-law callback can overflow exp before weighting it.
            def integrand(y):
                value = v.scalar(args[0](float(y)), "expectation callback")
                return value * np.exp(self.evaluate("log_density", np.asarray(y)))
            lower, upper = self.evaluate("support")
            return integrate.quad(integrand, lower, upper, **kwargs)[0]
        if self.affine_parameters is not None:
            shift, scale = self.affine_parameters
            if operation == "mean":
                return shift + scale*self.base.mean()
            if operation == "variance":
                return scale**2*self.base.variance()
            if operation == "moment":
                n = args[0]
                return sum(comb(n, k)*shift**(n-k)*scale**k*self.base.moment(k) for k in range(n+1))
        raise NotImplementedError(operation)
