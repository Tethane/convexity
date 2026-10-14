"""scikit-learn fits with copied state and explicit prediction capabilities."""

import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import _validation as v
from ..interfaces import Diagnostics, FittedModel


def fit_estimator(model, x, y, w):
    from sklearn.exceptions import ConvergenceWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(
            np.array(x), np.array(y), sample_weight=None if w is None else np.array(w)
        )
    failures = [
        warning
        for warning in caught
        if issubclass(warning.category, ConvergenceWarning)
    ]
    for warning in caught:
        if not issubclass(warning.category, ConvergenceWarning):
            warnings.warn(str(warning.message), warning.category, stacklevel=2)
    iterations = getattr(model, "n_iter_", None)
    return Diagnostics(
        backend=f"sklearn.{type(model).__name__}",
        converged=not failures,
        message="; ".join(str(w.message) for w in failures),
        iterations=None if iterations is None else int(np.max(iterations)),
    )


@dataclass(frozen=True)
class LinearAdapter:
    coefficients: Any
    intercept: Any
    output_shape: tuple[int, ...] = ()
    capabilities: frozenset[str] = frozenset({"mean"})

    def __post_init__(self):
        object.__setattr__(
            self, "coefficients", v.array(self.coefficients, "coefficients")
        )
        object.__setattr__(self, "intercept", v.array(self.intercept, "intercept"))

    @property
    def input_dimension(self):
        return self.coefficients.shape[0]

    def evaluate(self, operation, features, *, offset=None):
        return features @ self.coefficients + self.intercept


@dataclass(frozen=True)
class EstimatorAdapter:
    _estimator: Any = field(repr=False)
    input_dimension: int
    output_shape: tuple[int, ...]
    capabilities: frozenset[str]

    def __post_init__(self):
        object.__setattr__(self, "_estimator", deepcopy(self._estimator))

    def evaluate(self, operation, features, *, offset=None):
        if operation == "probability":
            return self._estimator.predict_proba(features)
        return np.asarray(self._estimator.predict(features)).reshape(
            (len(features),) + self.output_shape
        )


def linear(spec, x, y, w):
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

    options = dict(fit_intercept=spec.fit_intercept)
    penalty = spec.penalty
    if penalty == "elastic_net" and spec.l1_ratio in (0, 1):
        penalty = "ridge" if spec.l1_ratio == 0 else "lasso"
    # Normalize weights so scaling them never changes the penalty.
    normalized = None if w is None else (np.asarray(w) / w.sum()) * len(x)
    if penalty == "none":
        model = LinearRegression(**options)
    elif penalty == "ridge":
        model = Ridge(alpha=len(x) * spec.alpha, solver="svd", **options)
    elif penalty == "lasso":
        model = Lasso(
            alpha=spec.alpha, tol=spec.tolerance, max_iter=spec.maxiter, **options
        )
    else:
        model = ElasticNet(
            alpha=spec.alpha,
            l1_ratio=spec.l1_ratio,
            tol=spec.tolerance,
            max_iter=spec.maxiter,
            **options,
        )
    diagnostics = fit_estimator(model, x, y, normalized)
    coefficients = np.asarray(model.coef_).T
    # Some estimators collapse a one-column target; keep the supplied target rank.
    coefficients = coefficients.reshape((x.shape[1],) + y.shape[1:])
    intercept = np.broadcast_to(np.asarray(model.intercept_), y.shape[1:])
    adapter = LinearAdapter(coefficients, intercept, y.shape[1:])
    return FittedModel(adapter, diagnostics, len(x), coefficients, intercept)


def logistic(spec, x, y, w):
    from sklearn.linear_model import LogisticRegression

    # The default L2 penalty is supported at both the dependency floor and current release.
    model = LogisticRegression(
        C=spec.C,
        fit_intercept=spec.fit_intercept,
        solver="lbfgs",
        tol=spec.tolerance,
        max_iter=spec.maxiter,
    )
    diagnostics = fit_estimator(model, x, y, w)
    adapter = EstimatorAdapter(model, x.shape[1], (), frozenset({"probability"}))
    return FittedModel(
        adapter,
        diagnostics,
        len(x),
        np.asarray(model.coef_).T,
        model.intercept_,
        classes=model.classes_,
    )


def quantile(spec, x, y, w):
    from sklearn.linear_model import QuantileRegressor

    models, reports = [], []
    for level in spec.quantiles:
        model = QuantileRegressor(
            quantile=level,
            alpha=spec.alpha,
            fit_intercept=spec.fit_intercept,
            solver="highs",
        )
        reports.append(fit_estimator(model, x, y, w))
        models.append(model)
    coefficients = np.column_stack([m.coef_ for m in models])
    intercept = np.array([m.intercept_ for m in models])
    diagnostics = Diagnostics(
        "sklearn.QuantileRegressor",
        all(d.converged for d in reports),
        "; ".join(d.message for d in reports if d.message),
        max(d.iterations or 0 for d in reports),
    )
    adapter = LinearAdapter(coefficients, intercept, (), frozenset({"quantile"}))
    return FittedModel(
        adapter, diagnostics, len(x), coefficients, intercept, quantiles=spec.quantiles
    )


def trees(spec, x, y, w):
    from sklearn import ensemble, tree

    suffix = "Classifier" if spec.task == "classification" else "Regressor"
    options = dict(
        max_depth=spec.max_depth,
        min_samples_leaf=spec.min_samples_leaf,
        random_state=spec.seed,
    )
    if spec.kind == "decision_tree":
        model = getattr(tree, "DecisionTree" + suffix)(**options)
    else:
        options["n_estimators"] = spec.n_estimators
        if spec.kind == "gradient_boosting":
            options["learning_rate"] = spec.learning_rate
            if spec.task == "regression":
                options["loss"] = "squared_error"
            model = getattr(ensemble, "GradientBoosting" + suffix)(**options)
        else:
            name = "RandomForest" if spec.kind == "random_forest" else "ExtraTrees"
            model = getattr(ensemble, name + suffix)(n_jobs=1, **options)
    fit_y = y[:, 0] if y.ndim == 2 and y.shape[1] == 1 else y
    diagnostics = fit_estimator(model, x, fit_y, w)
    classes = model.classes_ if spec.task == "classification" else None
    caps = frozenset({"probability" if classes is not None else "mean"})
    adapter = EstimatorAdapter(model, x.shape[1], y.shape[1:], caps)
    return FittedModel(adapter, diagnostics, len(x), classes=classes)
