"""Public optional PyTorch namespace; importing this module does not import torch."""

from .builtins.tensor import (
    TensorDistribution, bernoulli, binomial, from_torch, independent, lognormal,
    mixture, multivariate_normal, normal, poisson, student_t, transformed,
)

__all__ = ["TensorDistribution", "from_torch", "normal", "student_t", "lognormal",
           "bernoulli", "binomial", "poisson", "multivariate_normal", "independent",
           "mixture", "transformed"]
