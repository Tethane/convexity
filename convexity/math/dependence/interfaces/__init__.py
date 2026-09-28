"""Public mathematical state and extension contracts for family 5."""

from .base import ContinuousMarginal, ConvergenceError, CopulaAdapter, CovarianceEstimator, Diagnostics
from .copulas import Copula, CopulaDistribution, CopulaFit
from .covariance import CovarianceEstimate, FactorCovariance, MatrixRepair, fit_covariance
from .tails import PeaksOverThresholdResult, StudentTResult, TailFit

__all__ = ["ContinuousMarginal", "ConvergenceError", "CopulaAdapter", "CovarianceEstimator",
           "Diagnostics", "Copula", "CopulaDistribution", "CopulaFit", "CovarianceEstimate",
           "FactorCovariance", "MatrixRepair", "fit_covariance", "PeaksOverThresholdResult",
           "StudentTResult", "TailFit"]
