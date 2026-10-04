# `convexity.math.learning`

[Reference index](README.md) · [Contract conventions](../public_contracts.md)

Regression and general predictive learning on supplied mathematical inputs.

from convexity.math import learning as learn
model = learn.Ridge(alpha=.1).fit(training_features, training_response)
means = model.predict(new_features)

No implicit data split, time axis, uncertainty law or feature scaling.

## `DecisionTree`

```python
DecisionTree(task: 'str' = 'regression', max_depth: 'int | None' = None, min_samples_leaf: 'int' = 1, n_estimators: 'int' = 100, learning_rate: 'float' = 0.1, seed: 'int' = 0) -> None
```

Single regression or classification tree with explicit depth and leaf-size constraints.

### `DecisionTree.fit`

```python
fit(self, features: 'ArrayLike', response: 'ArrayLike', *, sample_weight: 'ArrayLike | None' = None) -> 'FittedModel'
```

## `ElasticNet`

```python
ElasticNet(alpha: 'float' = 1.0, l1_ratio: 'float' = 0.5, fit_intercept: 'bool' = True, tolerance: 'float' = 1e-08, maxiter: 'int' = 10000, check_convergence: 'bool' = True) -> None
```

Linear regression with combined L1/L2 coefficient penalty; the intercept is unpenalized.

### `ElasticNet.fit`

```python
fit(self, features: 'ArrayLike', response: 'ArrayLike', *, sample_weight: 'ArrayLike | None' = None) -> 'FittedModel'
```

## `ExtraTrees`

```python
ExtraTrees(task: 'str' = 'regression', max_depth: 'int | None' = None, min_samples_leaf: 'int' = 1, n_estimators: 'int' = 100, learning_rate: 'float' = 0.1, seed: 'int' = 0) -> None
```

Randomized tree ensemble for the selected prediction task, with explicit seed and ensemble size.

### `ExtraTrees.fit`

```python
fit(self, features: 'ArrayLike', response: 'ArrayLike', *, sample_weight: 'ArrayLike | None' = None) -> 'FittedModel'
```

## `GeneralizedLinearModel`

```python
GeneralizedLinearModel(family: 'str' = 'gaussian', fit_intercept: 'bool' = True, tolerance: 'float' = 1e-08, maxiter: 'int' = 100, check_convergence: 'bool' = True) -> None
```

Gaussian/identity, Poisson/log, Gamma/log, or Binomial/logit.

Binomial response is a success count when trials is supplied, else 0/1.
Its predicted mean is always the probability per trial, not a count.

### `GeneralizedLinearModel.fit`

```python
fit(self, features: 'ArrayLike', response: 'ArrayLike', *, frequency_weights: 'ArrayLike | None' = None, trials: 'ArrayLike | None' = None, offset: 'ArrayLike | None' = None) -> 'FittedModel'
```

## `GradientBoosting`

```python
GradientBoosting(task: 'str' = 'regression', max_depth: 'int | None' = 3, min_samples_leaf: 'int' = 1, n_estimators: 'int' = 100, learning_rate: 'float' = 0.1, seed: 'int' = 0) -> None
```

Sequential tree boosting with explicit learning rate, depth and iteration count.

### `GradientBoosting.fit`

```python
fit(self, features: 'ArrayLike', response: 'ArrayLike', *, sample_weight: 'ArrayLike | None' = None) -> 'FittedModel'
```

## `Lasso`

```python
Lasso(alpha: 'float' = 1.0, l1_ratio: 'float' = 0.5, fit_intercept: 'bool' = True, tolerance: 'float' = 1e-08, maxiter: 'int' = 10000, check_convergence: 'bool' = True) -> None
```

Linear regression with an L1 coefficient penalty; no feature scaling is implicit.

### `Lasso.fit`

```python
fit(self, features: 'ArrayLike', response: 'ArrayLike', *, sample_weight: 'ArrayLike | None' = None) -> 'FittedModel'
```

## `LinearRegression`

```python
LinearRegression(penalty: 'str' = 'none', alpha: 'float' = 1.0, l1_ratio: 'float' = 0.5, fit_intercept: 'bool' = True, tolerance: 'float' = 1e-08, maxiter: 'int' = 10000, check_convergence: 'bool' = True) -> None
```

Weighted mean squared loss / 2 plus an unscaled coefficient penalty.

The intercept is optional and unpenalized. No feature scaling is implicit.
Each target in a matrix response is fit with its own coefficient penalty.

### `LinearRegression.fit`

```python
fit(self, features: 'ArrayLike', response: 'ArrayLike', *, sample_weight: 'ArrayLike | None' = None) -> 'FittedModel'
```

## `LogisticRegression`

```python
LogisticRegression(C: 'float' = 1.0, fit_intercept: 'bool' = True, tolerance: 'float' = 1e-08, maxiter: 'int' = 1000, check_convergence: 'bool' = True) -> None
```

Binary/multinomial L2 logistic learning; C uses scikit-learn's convention.

### `LogisticRegression.fit`

```python
fit(self, features: 'ArrayLike', response: 'ArrayLike', *, sample_weight: 'ArrayLike | None' = None) -> 'FittedModel'
```

## `NeuralClassifier`

```python
NeuralClassifier(hidden_layers: 'tuple[int, ...]' = (32, 32), activation: 'str' = 'tanh', learning_rate: 'float' = 0.01, maxiter: 'int' = 2000, tolerance: 'float' = 1e-05, seed: 'int' = 0, check_convergence: 'bool' = True) -> None
```

Feed-forward classifier with explicit architecture, training controls and random seed.

### `NeuralClassifier.fit`

```python
fit(self, features: 'ArrayLike', response: 'ArrayLike', *, sample_weight: 'ArrayLike | None' = None) -> 'FittedModel'
```

## `NeuralNetwork`

```python
NeuralNetwork(task: 'str' = 'regression', hidden_layers: 'tuple[int, ...]' = (32, 32), activation: 'str' = 'tanh', learning_rate: 'float' = 0.01, maxiter: 'int' = 2000, tolerance: 'float' = 1e-05, seed: 'int' = 0, check_convergence: 'bool' = True) -> None
```

Full-batch CPU float64 Adam; convergence checks the gradient infinity norm.

### `NeuralNetwork.fit`

```python
fit(self, features: 'ArrayLike', response: 'ArrayLike', *, sample_weight: 'ArrayLike | None' = None) -> 'FittedModel'
```

## `NeuralRegressor`

```python
NeuralRegressor(hidden_layers: 'tuple[int, ...]' = (32, 32), activation: 'str' = 'tanh', learning_rate: 'float' = 0.01, maxiter: 'int' = 2000, tolerance: 'float' = 1e-05, seed: 'int' = 0, check_convergence: 'bool' = True) -> None
```

Feed-forward mean regressor with explicit architecture, training controls and random seed.

### `NeuralRegressor.fit`

```python
fit(self, features: 'ArrayLike', response: 'ArrayLike', *, sample_weight: 'ArrayLike | None' = None) -> 'FittedModel'
```

## `Pipeline`

```python
Pipeline(transforms: 'Sequence[TransformFitter]', learner: 'Learner') -> None
```

Fit transformations and a learner on exactly the supplied training rows.

Weights apply to the learner; preprocessing uses unweighted training rows.
Callers needing weighted preprocessing should supply a custom transform.

### `Pipeline.fit`

```python
fit(self, features: 'ArrayLike', response: 'ArrayLike', **kwargs) -> 'FittedModel'
```

## `QuantileRegression`

```python
QuantileRegression(quantiles: 'tuple[float, ...]' = (0.5,), alpha: 'float' = 0.0, fit_intercept: 'bool' = True, check_convergence: 'bool' = True) -> None
```

Independent linear quantiles: weighted pinball loss + alpha*||beta||_1.

### `QuantileRegression.fit`

```python
fit(self, features: 'ArrayLike', response: 'ArrayLike', *, sample_weight: 'ArrayLike | None' = None) -> 'FittedModel'
```

## `RandomForest`

```python
RandomForest(task: 'str' = 'regression', max_depth: 'int | None' = None, min_samples_leaf: 'int' = 1, n_estimators: 'int' = 100, learning_rate: 'float' = 0.1, seed: 'int' = 0) -> None
```

Bagged tree ensemble for the selected prediction task, with explicit seed and ensemble size.

### `RandomForest.fit`

```python
fit(self, features: 'ArrayLike', response: 'ArrayLike', *, sample_weight: 'ArrayLike | None' = None) -> 'FittedModel'
```

## `Ridge`

```python
Ridge(alpha: 'float' = 1.0, l1_ratio: 'float' = 0.5, fit_intercept: 'bool' = True, tolerance: 'float' = 1e-08, maxiter: 'int' = 10000, check_convergence: 'bool' = True) -> None
```

Linear regression with an L2 coefficient penalty; no feature scaling is implicit.

### `Ridge.fit`

```python
fit(self, features: 'ArrayLike', response: 'ArrayLike', *, sample_weight: 'ArrayLike | None' = None) -> 'FittedModel'
```

## `Tree`

```python
Tree(task: 'str' = 'regression', kind: 'str' = 'decision_tree', max_depth: 'int | None' = None, min_samples_leaf: 'int' = 1, n_estimators: 'int' = 100, learning_rate: 'float' = 0.1, seed: 'int' = 0) -> None
```

Reusable tree-learning specification. task determines mean versus class-probability capabilities.

### `Tree.fit`

```python
fit(self, features: 'ArrayLike', response: 'ArrayLike', *, sample_weight: 'ArrayLike | None' = None) -> 'FittedModel'
```

## `elastic_net`

```python
elastic_net(features: 'ArrayLike', response: 'ArrayLike', *, alpha: 'float' = 1.0, l1_ratio: 'float' = 0.5, sample_weight: 'ArrayLike | None' = None, **kwargs: 'Unpack[ElasticNetOptions]') -> 'FittedModel'
```

Fit ElasticNet to features (observation,feature) and aligned response.

ElasticNet(alpha: 'float' = 1.0, l1_ratio: 'float' = 0.5, fit_intercept: 'bool' = True, tolerance: 'float' = 1e-08, maxiter: 'int' = 10000, check_convergence: 'bool' = True)
Return the checked fitted predictor; no feature scaling is implicit.
Optional settings: fit_intercept, tolerance, maxiter, check_convergence.

Keyword options:

```python
fit_intercept: bool
tolerance: float
maxiter: int
check_convergence: bool
```

## `generalized_linear_model`

```python
generalized_linear_model(features: 'ArrayLike', response: 'ArrayLike', *, frequency_weights: 'ArrayLike | None' = None, trials: 'ArrayLike | None' = None, offset: 'ArrayLike | None' = None, **kwargs: 'Unpack[GeneralizedLinearModelOptions]') -> 'FittedModel'
```

Fit GeneralizedLinearModel to features (observation,feature) and aligned response.

Gaussian/identity, Poisson/log, Gamma/log, or Binomial/logit.
Return the checked fitted predictor; no feature scaling is implicit.
Optional settings: family, fit_intercept, tolerance, maxiter, check_convergence.

Keyword options:

```python
family: str
fit_intercept: bool
tolerance: float
maxiter: int
check_convergence: bool
```

## `lasso`

```python
lasso(features: 'ArrayLike', response: 'ArrayLike', *, alpha: 'float' = 1.0, sample_weight: 'ArrayLike | None' = None, **kwargs: 'Unpack[LassoOptions]') -> 'FittedModel'
```

Fit Lasso to features (observation,feature) and aligned response.

Lasso(alpha: 'float' = 1.0, l1_ratio: 'float' = 0.5, fit_intercept: 'bool' = True, tolerance: 'float' = 1e-08, maxiter: 'int' = 10000, check_convergence: 'bool' = True)
Return the checked fitted predictor; no feature scaling is implicit.
Optional settings: l1_ratio, fit_intercept, tolerance, maxiter, check_convergence.

Keyword options:

```python
l1_ratio: float
fit_intercept: bool
tolerance: float
maxiter: int
check_convergence: bool
```

## `linear_regression`

```python
linear_regression(features: 'ArrayLike', response: 'ArrayLike', *, sample_weight: 'ArrayLike | None' = None, **kwargs: 'Unpack[LinearRegressionOptions]') -> 'FittedModel'
```

Fit LinearRegression to features (observation,feature) and aligned response.

Weighted mean squared loss / 2 plus an unscaled coefficient penalty.
Return the checked fitted predictor; no feature scaling is implicit.
Optional settings: penalty, alpha, l1_ratio, fit_intercept, tolerance, maxiter, check_convergence.

Keyword options:

```python
penalty: str
alpha: float
l1_ratio: float
fit_intercept: bool
tolerance: float
maxiter: int
check_convergence: bool
```

## `logistic_regression`

```python
logistic_regression(features: 'ArrayLike', response: 'ArrayLike', *, sample_weight: 'ArrayLike | None' = None, **kwargs: 'Unpack[LogisticRegressionOptions]') -> 'FittedModel'
```

Fit LogisticRegression to features (observation,feature) and aligned response.

Binary/multinomial L2 logistic learning; C uses scikit-learn's convention.
Return the checked fitted predictor; no feature scaling is implicit.
Optional settings: C, fit_intercept, tolerance, maxiter, check_convergence.

Keyword options:

```python
C: float
fit_intercept: bool
tolerance: float
maxiter: int
check_convergence: bool
```

## `neural_function`

```python
neural_function(module: 'Module', input_dimension: 'int', *, output_shape: 'tuple[int, ...]' = (), classes: 'ArrayLike | None' = None) -> 'FittedModel'
```

Snapshot a torch module as a deterministic CPU float64 predictor.

Regression modules return (n,m), m=1 for scalar outputs. Classification
modules return (n,c) logits in the explicitly supplied class order.

## `quantile_regression`

```python
quantile_regression(features: 'ArrayLike', response: 'ArrayLike', *, quantiles: 'tuple[float, ...]' = (0.5,), sample_weight: 'ArrayLike | None' = None, **kwargs: 'Unpack[QuantileRegressionOptions]') -> 'FittedModel'
```

Fit QuantileRegression to features (observation,feature) and aligned response.

Independent linear quantiles: weighted pinball loss + alpha*||beta||_1.
Return the checked fitted predictor; no feature scaling is implicit.
Optional settings: alpha, fit_intercept, check_convergence.

Keyword options:

```python
alpha: float
fit_intercept: bool
check_convergence: bool
```

## `ridge`

```python
ridge(features: 'ArrayLike', response: 'ArrayLike', *, alpha: 'float' = 1.0, sample_weight: 'ArrayLike | None' = None, **kwargs: 'Unpack[RidgeOptions]') -> 'FittedModel'
```

Fit Ridge to features (observation,feature) and aligned response.

Ridge(alpha: 'float' = 1.0, l1_ratio: 'float' = 0.5, fit_intercept: 'bool' = True, tolerance: 'float' = 1e-08, maxiter: 'int' = 10000, check_convergence: 'bool' = True)
Return the checked fitted predictor; no feature scaling is implicit.
Optional settings: l1_ratio, fit_intercept, tolerance, maxiter, check_convergence.

Keyword options:

```python
l1_ratio: float
fit_intercept: bool
tolerance: float
maxiter: int
check_convergence: bool
```

## `ConvergenceError`

```python
ConvergenceError(result: 'FittedModel') -> 'None'
```

Raised when a checked numerical method does not converge; result retains the attempted solution and diagnostics.

## `Diagnostics`

```python
Diagnostics(backend: 'str' = 'user', converged: 'bool' = True, message: 'str' = '', iterations: 'int | None' = None) -> None
```

Backend termination diagnostics. Inspect converged and message before using a fitted approximation; iterations may be unavailable.

## `FittedModel`

```python
FittedModel(adapter: 'PredictionAdapter', diagnostics: 'Diagnostics' = <factory>, nobs: 'int | None' = None, coefficients: 'ArrayLike | None' = None, intercept: 'ArrayLike | None' = None, classes: 'ArrayLike | None' = None, quantiles: 'ArrayLike | None' = None, loss_history: 'ArrayLike | None' = None, requires_offset: 'bool' = False) -> 'None'
```

A checked predictor. Capabilities state exactly which outputs exist.

Adapters receive (n,d) and return (n,)+output_shape for means, (n,c)
for probabilities, or (n,q) for quantiles. All leading axes are restored.

### `FittedModel.predict`

```python
predict(self, features: 'ArrayLike', *, offset: 'ArrayLike | None' = None) -> 'NDArray[np.generic]'
```

Mean, class label, or the declared quantile vector, in that order.

### `FittedModel.predict_class`

```python
predict_class(self, features: 'ArrayLike', *, offset: 'ArrayLike | None' = None) -> 'NDArray[np.generic]'
```

### `FittedModel.predict_mean`

```python
predict_mean(self, features: 'ArrayLike', *, offset: 'ArrayLike | None' = None) -> 'FloatArray'
```

### `FittedModel.predict_proba`

```python
predict_proba(self, features: 'ArrayLike', *, offset: 'ArrayLike | None' = None) -> 'FloatArray'
```

### `FittedModel.predict_quantiles`

```python
predict_quantiles(self, features: 'ArrayLike', *, offset: 'ArrayLike | None' = None) -> 'FloatArray'
```

Return the fitted levels in order; never sort away quantile crossings.

## `Learner`

Fit feature rows and scalar/vector responses into a FittedModel; optional training settings belong to the implementation.

### `Learner.fit`

```python
fit(self, features: 'ArrayLike', response: 'ArrayLike') -> "'FittedModel'"
```

## `PredictionAdapter`

Prediction extension with input dimension, output shape and capabilities; evaluate returns aligned observation rows.

### `PredictionAdapter.evaluate`

```python
evaluate(self, operation: 'str', features: 'ArrayLike', *, offset: 'ArrayLike | None' = None) -> 'ArrayLike'
```

## `fit`

```python
fit(features: 'ArrayLike', response: 'ArrayLike', learner: 'Learner', **kwargs: 'object') -> 'FittedModel'
```

Fit a user or built-in learner and check every declared output on training rows.
