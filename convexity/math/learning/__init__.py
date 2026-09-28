"""Regression and general predictive learning on supplied mathematical inputs.

    from convexity.math import learning as learn
    model = learn.Ridge(alpha=.1).fit(training_features, training_response)
    means = model.predict(new_features)

    No implicit data split, time axis, uncertainty law or feature scaling.
"""

from . import builtins, interfaces
from .builtins import (
    DecisionTree, ElasticNet, ExtraTrees, GeneralizedLinearModel, GradientBoosting,
    Lasso, LinearRegression, LogisticRegression, NeuralClassifier, NeuralNetwork,
    NeuralRegressor, Pipeline, QuantileRegression, RandomForest, Ridge, Tree,
    elastic_net, generalized_linear_model, lasso, linear_regression,
    logistic_regression, neural_function, quantile_regression, ridge,
)
from .interfaces import ConvergenceError, Diagnostics, FittedModel, Learner, PredictionAdapter, fit

__all__ = ["builtins", "interfaces", "DecisionTree", "ElasticNet", "ExtraTrees",
           "GeneralizedLinearModel", "GradientBoosting", "Lasso", "LinearRegression",
           "LogisticRegression", "NeuralClassifier", "NeuralNetwork", "NeuralRegressor",
           "Pipeline", "QuantileRegression", "RandomForest", "Ridge", "Tree", "elastic_net",
           "generalized_linear_model", "lasso", "linear_regression", "logistic_regression",
           "neural_function", "quantile_regression", "ridge", "ConvergenceError", "Diagnostics",
           "FittedModel", "Learner", "PredictionAdapter", "fit"]
