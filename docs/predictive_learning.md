# Regression and general predictive learning

Family 6 learns conditional means, class probabilities and conditional quantiles
from supplied feature/response arrays. It fits reusable predictors; family 3
estimates parameters and their uncertainty. Neither interface acquires data or
chooses training/test rows for the caller.

```python
from convexity.math import learning as learn, transforms as tx

model = learn.Pipeline((tx.Standardize(),), learn.Ridge(alpha=.1)).fit(X_train, y_train)
conditional_means = model.predict(X_test)

classifier = learn.LogisticRegression(C=1.).fit(X_train, labels)
probabilities = classifier.predict_proba(X_test)  # columns follow classifier.classes

quantiles = learn.QuantileRegression((.1, .5, .9)).fit(X_train, y_train)
conditional_quantiles = quantiles.predict_quantiles(X_test)
```

Run the [generated-data problem book](../examples/predictive_learning.py),
or open the [worked notebook](../notebooks/predictive_learning.ipynb).

## Finite catalog

| Tools | Outputs and fitting backend |
|---|---|
| `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet` | Scalar/vector conditional means; scikit-learn least squares and penalized fits |
| `GeneralizedLinearModel` | Gaussian/identity, Poisson/log, Gamma/log and binomial/logit conditional means; statsmodels IRLS |
| `LogisticRegression` | Binary/multinomial class probabilities and labels; scikit-learn L2 logistic regression |
| `QuantileRegression` | One or more independently fitted linear conditional quantiles; scikit-learn/HiGHS |
| `DecisionTree`, `RandomForest`, `ExtraTrees` | Scalar/vector means or single-target class probabilities; scikit-learn |
| `GradientBoosting` | Scalar means or single-target class probabilities; scikit-learn |
| `NeuralRegressor`, `NeuralClassifier`, `NeuralNetwork` | Dense neural means or class probabilities; optional PyTorch |
| `neural_function` | A snapshot of a supplied PyTorch module; no fitting |
| `Pipeline`, `Learner`, `PredictionAdapter`, `FittedModel`, `fit` | Training-only feature composition and user-written tools |

Convenience functions include `linear_regression`, `ridge`, `lasso`, `elastic_net`,
`generalized_linear_model`, `logistic_regression` and `quantile_regression`.
`Tree(kind=..., task=...)` and `NeuralNetwork(task=...)` expose the configurable
base specifications. Public names are also available from `builtins`/`interfaces`.

This catalog does not include automatic model selection, cross-validation,
categorical feature encoding, missing-value imputation, streaming updates, survival
models, custom GLM links, neural GPU/minibatch training, multi-label classification
or automatic probability calibration. Kernels/GPs and time-series forecasts belong
to later families. These models do not create predictive distributions or confidence
intervals from point predictions. Quantile outputs do not define a joint law.

## Three layers and ownership

`interfaces/` defines prediction semantics, diagnostics and structural extension
protocols. `_backends/` adapts scikit-learn, statsmodels and PyTorch. `builtins/`
contains immutable fitting specifications and convenient functions. Importing
`learning` loads NumPy but no numerical backend, application or tensor stack.

Every `.fit` returns new state. Built-in linear/GLM/quantile fits retain copied,
read-only coefficient arrays. Tree/logistic adapters own private deep copies of
fitted estimators. Neural adapters deep-copy their module, move it to CPU float64
and enter evaluation mode. Original training arrays and supplied modules can be
changed without altering these predictions. Backend objects remain private
implementation details. This is ordinary Python ownership, not tamper-proof storage.

Public numeric state and prediction arrays are copied and read-only. User adapters
own their own state and determinism. Fitters never mutate input arrays. The caller
must keep feature columns in the same order at prediction time.

## Shapes and prediction meaning

| Value | Shape / meaning |
|---|---|
| Training features | Finite real `(n,d)`, with `n,d >= 1`; univariate features use `(n,1)` |
| Regression response | `(n,)` or `(n,m)` where supported; `(n,1)` retains its trailing output axis |
| GLM / quantile / gradient boosting response | Scalar `(n,)` only |
| Class labels | `(n,)` integer-valued numeric labels or nonempty Unicode strings; at least two positive-weight classes |
| Prediction features | `(...,d)`, including a vector `(d,)` and empty leading dimensions |
| Mean predictions | `(...)` for a scalar target; `(...,m)` for a vector target |
| Class probabilities | `(...,c)`, columns in the stored `classes` order; rows sum to one |
| Predicted class | `(...)`, maximum-probability label; ties use the first stored class |
| Quantile predictions | `(...,q)`, including a trailing axis for a single fitted quantile |
| Linear coefficients / intercept | `(d,)` / `()` or `(d,m)` / `(m,)` |
| Quantile coefficients / intercept | `(d,q)` / `(q,)` |
| Logistic coefficients / intercept | `(d,1)` / `(1,)` for binary log odds; `(d,c)` / `(c,)` for multinomial logits |

Inputs reject NaNs, infinities, complex values, booleans, numeric strings as
features/responses, empty training sets and mismatched rows. String **class labels**
are supported; object arrays and boolean labels require explicit conversion.
Labels produced by built-in learners are sorted, as in scikit-learn. A supplied
`neural_function` or custom adapter retains the explicitly declared class order.

`model.capabilities` is a frozen set containing `mean`, `probability`, and/or
`quantile`. `predict_mean`, `predict_proba`, `predict_class` and `predict_quantiles`
check capability, dimensions, finiteness and output shape. Unsupported outputs
raise `NotImplementedError`. `predict` chooses a mean if available, otherwise a
class label, otherwise the fitted quantile vector. A binomial GLM therefore
returns success probabilities from `predict`; logistic classifiers return labels.

## Linear regression and regularization

The fitted map is `x @ coefficients + intercept`. The default fits an unpenalized
intercept; `fit_intercept=False` sets it to zero. Features are not standardized.
Constant columns and rank deficiency are accepted by ordinary/ridge regression;
least squares uses the backend's minimum-norm solution when coefficients are not
identified. No coefficient covariance is inferred by this family.

With normalized positive observation weights `w_i` summing to one, each target
minimizes

```text
sum_i w_i * (y_i - b - x_i @ beta)^2 / 2
    + alpha * [r * ||beta||_1 + (1-r) * ||beta||_2^2 / 2].
```

Ridge uses `r=0`, lasso `r=1`, and elastic net its supplied `l1_ratio`. The latter's
endpoints use the ridge/lasso solver directly. `alpha > 0`; use `LinearRegression`
for an unpenalized fit. Matrix targets use separate coefficient penalties, with
no multi-task/group sparsity penalty. The ridge adapter scales scikit-learn's
ridge parameter to give this same mean-loss convention. With no intercept or
weights, coefficients agree with family 3's corresponding regularized fit.

`sample_weight` is a finite, nonnegative `(n,)` array with positive total weight.
Zero-weight rows are omitted before fitting and class discovery. Multiplying
all weights by a constant leaves linear, quantile and neural objectives unchanged.
`nobs` records positive-weight training rows, not a replicated/effective count.

## GLMs, counts and offsets

```python
counts = learn.GeneralizedLinearModel('poisson').fit(X, y_count, offset=log_exposure)
future_counts = counts.predict(X_new, offset=new_log_exposure)

binomial = learn.GeneralizedLinearModel('binomial').fit(X, successes, trials=trials)
p_success = binomial.predict_mean(X_new)
expected_successes = new_trials * p_success
```

The linear predictor is `eta = intercept + X @ coefficients + offset`. Gaussian
means are `eta`; Poisson and Gamma means are `exp(eta)`; binomial means are
`expit(eta)`. These fixed links are the finite GLM catalog. Prediction does not
clip extrapolated means or silently cap an overflowing log link.

Gaussian responses can be any finite real values. Poisson responses are
nonnegative integer counts; Gamma responses are strictly positive. A binomial
response is 0/1 by default. With a supplied positive-integer `(n,)` `trials`,
responses are integer successes between zero and trials. Internally this fits
the successes/failures representation. Proportions must be converted to the
intended counts explicitly. Binomial `predict_proba` returns `[1-p, p]` for a
single trial with `classes=[0,1]`; it is not a distribution over aggregate counts.

GLMs accept `frequency_weights`, nonnegative integer replication counts, instead
of generic `sample_weight`. Zero-frequency rows are removed. IRLS requires full
column rank and more retained rows than coefficients, including an inserted
intercept. Avoid an explicit constant feature when also fitting an intercept.
Separation in a binomial fit is reported as failed convergence.

Training offsets have shape `(n,)`. A fit using an offset **requires** an explicit
offset on every prediction; it never reuses training offsets or assumes new
exposure. Prediction offsets match all leading feature axes, without scalar
broadcasting. A GLM fitted without offsets can still accept an explicit new
offset. Other learners reject offsets. The caller computes log exposure explicitly.

## Logistic probability models

`LogisticRegression` uses scikit-learn's L2/LBFGS binary or multinomial model.
`C > 0` is scikit-learn's inverse regularization parameter, not linear regression's
normalized `alpha`. Its sample weights use the backend's summed-weight convention;
integer weights match replicated rows and scaling all weights changes the relative
penalty. Binary probabilities use a sigmoid of the single logit; multinomial
probabilities use softmax across logits. The intercept is unpenalized.

Probability output is the fitted model's conditional estimate. The interface
does not claim calibration on held-out data or infer class prevalence outside
the supplied training population.

## Quantiles, trees and ensembles

`QuantileRegression(quantiles=(.1,.5,.9), alpha=0)` independently minimizes weighted
mean pinball loss plus `alpha * ||beta||_1`. Quantiles must be unique, strictly
increasing, and inside `(0,1)`. The optional intercept is unpenalized. Only the
fitted levels are available. Independent quantiles may cross, especially under
extrapolation; output is never sorted or represented as an automatically valid
distribution or interval. The median is not labeled a conditional mean.

Trees and ensembles use squared-error regression or the backend's classification
criterion. Their probability estimates are leaf frequencies, averaged tree
probabilities, or boosting probabilities as appropriate. Forests/extra trees
support vector regression; gradient boosting accepts a scalar target.

`max_depth`, `min_samples_leaf` and `seed` configure all tree tools.
`n_estimators` configures ensembles; `learning_rate` configures gradient boosting.
Random forests bootstrap by default; extra trees do not. Forest workers use
`n_jobs=1`. Explicit integer seeds default to zero. Repeated fits with the same
inputs and library version are reproducible. These are finite fits, not numerical
convergence or optimality certificates.

## Neural learning and supplied neural functions

Install `pip install 'convexity[neural]'` (the existing `autodiff` extra also supplies
PyTorch). Neural dependencies are only imported on use.

```python
network = learn.NeuralRegressor(hidden_layers=(32, 16), activation='tanh', seed=7,
                               maxiter=4000, tolerance=1e-4).fit(X, y)
means = network.predict(X_new)

# A supplied module returns (n,1); it is copied, not trained by this function.
function = learn.neural_function(torch.nn.Linear(X.shape[1], 1), X.shape[1])
# Classification modules return logits in the supplied class order:
classifier = learn.neural_function(module, input_dimension=3, classes=['low', 'high'])
```

Dense learners support `tanh`, `relu` and `sigmoid` hidden activations. An empty
hidden-layer tuple is a linear output layer. Regression outputs are linear;
classification outputs are logits with softmax at prediction. Full-batch Adam
trains on CPU float64 using weighted squared error / 2 averaged over target
coordinates, or weighted cross-entropy. Features/targets are not scaled implicitly.

Convergence means the training objective's gradient infinity norm is at most
`tolerance`; it does not certify generalization or a global minimum. `maxiter`
counts Adam updates. The read-only `loss_history` includes initial and final losses.
An iteration limit produces `ConvergenceError` unless `check_convergence=False`.
Nonfinite losses/gradients raise `FloatingPointError`. Initialization preserves the
caller's CPU RNG state and does not seed GPU generators.

`neural_function` deep-copies any compatible `torch.nn.Module`, including parameters
and buffers, without modifying the original's training mode or gradients. It runs
in evaluation mode and requires `(n,m)` outputs, even `(n,1)` for scalar regression.
Use `output_shape=(m,)` for vector means; provide `classes` for logits. Inputs and
outputs use the NumPy family contract; this predictor does not expose an autograd
graph. Custom modules own their behavior and any stochastic operations in `forward`.

## Composition, extension and diagnostics

`Pipeline((transform_fitters,...), learner)` fits family 4's row-preserving
transforms on training features, then fits the learner in those coordinates.
Predictions reuse both fitted stages. Scaling/PCA use the supplied training rows
without response information. Learner weights apply only to the learner;
preprocessing is unweighted, including any zero-weight rows supplied to the
pipeline. Select rows first when they must also be excluded from preprocessing.
Row-dropping lag/difference operators need explicit response alignment beforehand.

Pipeline state is inspectable at `result.adapter.transform` and
`result.adapter.model`. The outer result does not expose transformed-coordinate
coefficients as if they belonged to the original feature space.

```python
import numpy as np

class ConstantAdapter:
    input_dimension = 2
    output_shape = ()
    capabilities = frozenset({'mean'})

    def evaluate(self, operation, features, *, offset=None):
        return np.full(len(features), 3.)

class CustomLearner:
    def fit(self, features, response):
        return learn.FittedModel(ConstantAdapter())

custom = learn.fit([[1., 2.], [3., 4.]], [2., 4.], CustomLearner())
```

No registry or inheritance is required. A `PredictionAdapter` receives a read-only
`(n,d)` array and returns `(n,)+output_shape` means, `(n,c)` probabilities or `(n,q)`
quantiles. `FittedModel` handles leading axes, empty predictions and output checks.
Provide `classes`/`quantiles` when declaring those capabilities. Custom offset
adapters declare `supports_offset=True`. Custom learners define target/extra-option
semantics; `fit` checks the returned model, feature dimension and all declared
predictions on training features. It does not override custom convergence policies.

`Diagnostics` records backend, convergence status, message and available iteration
count. Iterative built-ins raise `ConvergenceError`, retaining the usable failed
model as `.result`, when their backend or explicit convergence criterion fails.
`check_convergence=False` returns that same state for inspection. Backend exceptions
before a finite fitted state exists propagate; unsupported or malformed inputs
raise before fitting. No automatic retry changes the requested model.

## Verification

```bash
python -m pytest -q tests/math/learning
python -m pyright --pythonpath "$(command -v python)"
python examples/predictive_learning.py
python examples/predictive_learning.py --neural
```

Tests check weighted normal equations, ridge shrinkage and elastic-net soft
thresholding; agreement with family 3; independent vector targets; empirical and
conditional quantiles; grouped/expanded binomial equivalence; GLM links, exposures
and generated-parameter recovery; logistic/softmax probabilities; nonlinear tree
recovery; neural recovery, gradients, RNG isolation and module snapshots; pipeline
training isolation; convergence failures, invalid domains, ownership and custom
adapters. Optional neural tests require a working PyTorch installation.

Verified: all 163 family tests pass with current dependencies, including eight
PyTorch tests. The 155 base tests also pass with NumPy 1.26.4, SciPy 1.15.3,
scikit-learn 1.6.1 and statsmodels 0.14.4; the optional neural tests are skipped
in that environment. The script (including `--neural`) and every notebook cell
execute successfully. On the local machine, `env -u LD_LIBRARY_PATH` avoids an
incompatible `/opt/libtorch` override when running PyTorch checks.

Backend references: [scikit-learn linear models](https://scikit-learn.org/stable/modules/linear_model.html),
[logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html),
[quantile regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html),
[statsmodels GLM](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.html),
[tree ensembles](https://scikit-learn.org/stable/modules/ensemble.html),
[PyTorch optimizers](https://docs.pytorch.org/docs/stable/optim.html).
