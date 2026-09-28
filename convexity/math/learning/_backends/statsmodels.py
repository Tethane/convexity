"""statsmodels IRLS fitting; prediction uses copied coefficients and link state."""

from dataclasses import dataclass
import warnings

import numpy as np

from .sklearn import LinearAdapter
from ..interfaces import Diagnostics, FittedModel


@dataclass(frozen=True)
class GLMAdapter(LinearAdapter):
    family: str = "gaussian"
    supports_offset: bool = True

    def evaluate(self, operation, features, *, offset=None):
        from scipy.special import expit

        eta = features @ self.coefficients + self.intercept
        if offset is not None:
            eta = eta + offset
        if self.family == "binomial":
            mean = expit(eta)
        elif self.family == "gaussian":
            mean = eta
        else:
            mean = np.exp(eta)
        if operation == "probability":
            return np.column_stack([1 - mean, mean])
        return mean


def glm(spec, x, y, frequency_weights, trials, offset):
    from statsmodels.genmod.generalized_linear_model import GLM
    from statsmodels.genmod import families as sm_families
    from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationWarning

    families = {"gaussian": sm_families.Gaussian, "poisson": sm_families.Poisson,
                "gamma": sm_families.Gamma, "binomial": sm_families.Binomial}
    family = families[spec.family](link=sm_families.links.Log()) if spec.family == "gamma" else families[spec.family]()
    design = np.column_stack([np.ones(len(x)), x]) if spec.fit_intercept else x
    if len(design) <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("GLM requires a full-rank design and more positive-frequency rows than coefficients")
    endog = np.column_stack([y, trials - y]) if trials is not None else y
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = GLM(endog, design, family=family, freq_weights=frequency_weights,
                     offset=offset, missing="raise").fit(maxiter=spec.maxiter, tol=spec.tolerance)
    failures = [warning for warning in caught if issubclass(warning.category, ConvergenceWarning)
                or (spec.family == "binomial" and issubclass(warning.category, PerfectSeparationWarning))]
    for warning in caught:
        if warning not in failures:
            warnings.warn(str(warning.message), warning.category, stacklevel=2)
    converged = bool(result.converged) and not failures
    diagnostics = Diagnostics(f"statsmodels.GLM.{spec.family}", converged,
                              "; ".join(str(w.message) for w in failures) or
                              ("" if converged else "IRLS iteration limit reached"),
                              int(result.fit_history["iteration"]))
    coefficients = result.params[1:] if spec.fit_intercept else result.params
    intercept = result.params[0] if spec.fit_intercept else 0.
    caps = frozenset({"mean", "probability"} if spec.family == "binomial" else {"mean"})
    adapter = GLMAdapter(coefficients, intercept, capabilities=caps, family=spec.family)
    return FittedModel(adapter, diagnostics, len(x), coefficients, intercept,
                       classes=[0, 1] if spec.family == "binomial" else None,
                       requires_offset=offset is not None)
