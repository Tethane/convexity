"""Statistical estimation: checked problems, library-backed tools and built-ins.

    from convexity.math import distributions as dist, estimation as est
    model = est.likelihood(lambda p: dist.normal(p[0], 1), [1., 2., 3.])
    result = est.maximum_likelihood(model, [0.])
    est.wald_interval(result)

No data acquisition, predictive model stack or implicit parameter transforms.
"""

from . import builtins, interfaces
from .builtins import (
    ElasticNetPenalty,
    IndependentPrior,
    LeastSquares,
    LinearLeastSquares,
    RegularizedLinear,
    RobustLinear,
    ScipyMinimizer,
    bootstrap,
    delta_method,
    elastic_net,
    elastic_net_penalty,
    gaussian_prior,
    l1_penalty,
    l2_penalty,
    laplace_interval,
    laplace_prior,
    lasso,
    least_squares,
    likelihood,
    linear_least_squares,
    maximum_a_posteriori,
    maximum_likelihood,
    minimize_estimate,
    regularized_linear,
    ridge,
    robust_linear,
    wald_interval,
)
from .interfaces import (
    BootstrapResult,
    ConvergenceError,
    Diagnostics,
    Estimate,
    EstimationProblem,
    Estimator,
    Interval,
    LeastSquaresProblem,
    Likelihood,
    LinearConstraint,
    LinearProblem,
    NonlinearConstraint,
    ObjectiveProblem,
    ParameterSpace,
    fit,
)

__all__ = [
    "builtins",
    "interfaces",
    "ElasticNetPenalty",
    "IndependentPrior",
    "LeastSquares",
    "LinearLeastSquares",
    "RegularizedLinear",
    "RobustLinear",
    "ScipyMinimizer",
    "bootstrap",
    "delta_method",
    "elastic_net",
    "elastic_net_penalty",
    "gaussian_prior",
    "l1_penalty",
    "l2_penalty",
    "laplace_interval",
    "laplace_prior",
    "lasso",
    "least_squares",
    "likelihood",
    "linear_least_squares",
    "maximum_a_posteriori",
    "maximum_likelihood",
    "minimize_estimate",
    "regularized_linear",
    "ridge",
    "robust_linear",
    "wald_interval",
    "BootstrapResult",
    "ConvergenceError",
    "Diagnostics",
    "Estimate",
    "EstimationProblem",
    "Estimator",
    "Interval",
    "LeastSquaresProblem",
    "Likelihood",
    "LinearConstraint",
    "LinearProblem",
    "NonlinearConstraint",
    "ObjectiveProblem",
    "ParameterSpace",
    "fit",
]

from .builtins.objectives import LogDensityLaw

__all__ += ["LogDensityLaw"]
