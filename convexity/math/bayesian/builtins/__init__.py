"""Conjugate and library-backed Bayesian algorithms."""

from .conjugate import (
    BetaBinomial,
    GammaPoisson,
    GaussianLinearRegression,
    NormalInverseGamma,
    NormalMean,
    beta,
    gamma,
    inverse_gamma,
)
from .inference import MCMC, VariationalInference
from .prediction import posterior_predictive
from .sequential import JointParticleFilter

__all__ = [
    "BetaBinomial",
    "GammaPoisson",
    "GaussianLinearRegression",
    "NormalInverseGamma",
    "NormalMean",
    "beta",
    "gamma",
    "inverse_gamma",
    "MCMC",
    "VariationalInference",
    "posterior_predictive",
    "JointParticleFilter",
]
