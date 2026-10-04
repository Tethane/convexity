"""Bayesian inference and parameter uncertainty (family 11).

Conjugate inference uses NumPy/SciPy; MCMC and VI use optional PyMC, and joint
sequential inference composes the state-space particles backend. Heavy libraries
are imported only when needed. Stochastic operations require an explicit RNG.
"""
from . import builtins, interfaces
from .builtins import (BetaBinomial, GammaPoisson, GaussianLinearRegression, JointParticleFilter,
                       MCMC, NormalInverseGamma, NormalMean, VariationalInference,
                       beta, gamma, inverse_gamma, posterior_predictive)
from .interfaces import (BayesianProblem, InferenceDiagnostics, InferenceMethod, JointFilterResult,
                         PosteriorSamples, PyMCModel, StateParameterModel, infer)

__all__ = ['builtins', 'interfaces', 'BetaBinomial', 'GammaPoisson', 'GaussianLinearRegression',
           'JointParticleFilter', 'MCMC', 'NormalInverseGamma', 'NormalMean', 'VariationalInference',
           'beta', 'gamma', 'inverse_gamma', 'posterior_predictive', 'BayesianProblem',
           'InferenceDiagnostics', 'InferenceMethod', 'JointFilterResult', 'PosteriorSamples',
           'PyMCModel', 'StateParameterModel', 'infer']
