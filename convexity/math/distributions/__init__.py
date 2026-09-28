"""Probability distributions and conditional laws, independent of applications.

    from convexity.math import distributions as dist
    law = dist.normal(loc=2., scale=3.)
    law.quantile([.05, .5, .95])

SciPy and NumPy laws return checked arrays. Optional tensor laws preserve native
PyTorch tensors and autograd graphs. No data access, estimation or model fitting.
"""

from . import builtins, interfaces, tensor
from .builtins import (
    affine, bernoulli, binomial, empirical, exponential, independent, lognormal,
    mixture, multivariate_normal, multivariate_student_t, normal, poisson,
    student_t, transformed,
)
from .interfaces import Distribution, DistributionAdapter, Measure, MonotoneTransform

__all__ = ["builtins", "interfaces", "tensor", "Distribution", "DistributionAdapter",
           "Measure", "MonotoneTransform", "normal", "student_t", "lognormal",
           "bernoulli", "binomial", "poisson", "multivariate_normal",
           "multivariate_student_t", "empirical", "independent", "mixture",
           "transformed", "affine", "exponential"]
