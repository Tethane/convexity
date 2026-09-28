"""Transforms, decomposition and factor geometry on supplied arrays.

    from convexity.math import transforms as tx
    fitted = tx.Pipeline((tx.Standardize(), tx.PCA(2))).fit(training_matrix)
    scores = fitted.transform(new_matrix)
    approximation = fitted.reconstruct(scores)

No calendars, data acquisition, forecasting or trading dependencies.
"""

from . import builtins, interfaces
from .builtins import (
    Difference, FactorAnalysis, FittedPipeline, FourierFeatures, LagFeatures, MinMaxScale,
    PCA, Pipeline, RobustScale, SeasonalFeatures, Standardize, Whitening, align_subspaces,
    difference, factor_analysis, fourier_features, lag_features, minmax_scale, pca,
    principal_angles, robust_scale, seasonal_features, standardize, subspace, whitening,
)
from .interfaces import (
    AffineTransform, Alignment, ConvergenceError, Diagnostics, DifferenceResult,
    FactorAnalysisResult, FeatureMatrix, FittedScaling, GaussianFactorModel, PCAResult,
    Subspace, Transform, TransformFitter, checked_transform, fit,
)

__all__ = ["builtins", "interfaces", "Difference", "FactorAnalysis", "FittedPipeline",
           "FourierFeatures", "LagFeatures", "MinMaxScale", "PCA", "Pipeline", "RobustScale",
           "SeasonalFeatures", "Standardize", "Whitening", "align_subspaces", "difference",
           "factor_analysis", "fourier_features", "lag_features", "minmax_scale", "pca",
           "principal_angles", "robust_scale", "seasonal_features", "standardize", "subspace",
           "whitening", "AffineTransform", "Alignment", "ConvergenceError", "Diagnostics",
           "DifferenceResult", "FactorAnalysisResult", "FeatureMatrix", "FittedScaling",
           "GaussianFactorModel", "PCAResult", "Subspace", "Transform", "TransformFitter",
           "checked_transform", "fit"]
