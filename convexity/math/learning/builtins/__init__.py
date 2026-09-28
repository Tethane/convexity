"""Reusable predictive-learning specifications and convenience functions."""

from .composition import Pipeline
from .glm import GeneralizedLinearModel, generalized_linear_model
from .linear import (
    ElasticNet, Lasso, LinearRegression, LogisticRegression, QuantileRegression,
    Ridge, elastic_net, lasso, linear_regression, logistic_regression,
    quantile_regression, ridge,
)
from .neural import NeuralClassifier, NeuralNetwork, NeuralRegressor, neural_function
from .trees import DecisionTree, ExtraTrees, GradientBoosting, RandomForest, Tree

__all__ = ["Pipeline", "GeneralizedLinearModel", "generalized_linear_model", "ElasticNet",
           "Lasso", "LinearRegression", "LogisticRegression", "QuantileRegression", "Ridge",
           "elastic_net", "lasso", "linear_regression", "logistic_regression", "quantile_regression",
           "ridge", "NeuralClassifier", "NeuralNetwork", "NeuralRegressor", "neural_function",
           "DecisionTree", "ExtraTrees", "GradientBoosting", "RandomForest", "Tree"]
