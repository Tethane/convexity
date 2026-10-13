# `convexity.math.estimation`

[Reference index](README.md) · [Contract conventions](../public_contracts.md)

Statistical estimation: checked problems, library-backed tools and built-ins.

    from convexity.math import distributions as dist, estimation as est
    model = est.likelihood(lambda p: dist.normal(p[0], 1), [1., 2., 3.])
    result = est.maximum_likelihood(model, [0.])
    est.wald_interval(result)

No data acquisition, predictive model stack or implicit parameter transforms.

## `ElasticNetPenalty`

```python
ElasticNetPenalty(alpha: 'float' = 1.0, l1_ratio: 'float' = 0.5, center: 'ArrayLike' = 0.0, weights: 'ArrayLike' = 1.0) -> None
```

alpha * sum weights * [ratio*|theta-center| + (1-ratio)*(theta-center)²/2].

## `IndependentPrior`

```python
IndependentPrior(family: 'str' = 'normal', loc: 'ArrayLike' = 0.0, scale: 'ArrayLike' = 1.0) -> None
```

Independent product of scalar parameter priors, in declared parameter order.

## `LeastSquares`

```python
LeastSquares(method: 'str' = 'trf', loss: 'str' = 'linear', f_scale: 'float' = 1.0, absolute_sigma: 'bool' = False, tolerance: 'float' = 1e-09, max_evaluations: 'int' = 2000) -> None
```

Estimate parameters by minimizing supplied residuals; initial parameters, bounds and covariance interpretation are explicit.

## `LinearLeastSquares`

```python
LinearLeastSquares(covariance_type: 'str' = 'nonrobust') -> None
```

Fit a linear design to observations, retaining coefficient covariance and fit diagnostics when available.

## `RegularizedLinear`

```python
RegularizedLinear(alpha: 'float' = 1.0, l1_ratio: 'float' = 0.0, tolerance: 'float' = 1e-08, maxiter: 'int' = 10000) -> None
```

SSE/(2n) + alpha * [l1_ratio*||theta||_1 + (1-l1_ratio)*||theta||²/2].

Every supplied column is penalized, including an explicit constant column.
No intercept is inserted, centered, or exempted; no features are rescaled.

## `RobustLinear`

```python
RobustLinear(loss: 'str' = 'huber', tuning: 'float | None' = None, covariance_type: 'str' = 'H1', tolerance: 'float' = 1e-08, maxiter: 'int' = 200) -> None
```

Fit a linear response with the selected robust loss; robustness does not remove identification requirements.

## `ScipyMinimizer`

```python
ScipyMinimizer(method: 'str' = 'auto', tolerance: 'float' = 1e-08, maxiter: 'int' = 1000) -> None
```

Local optimization. auto selects SLSQP for bounds/constraints, L-BFGS-B otherwise.

hessian in ObjectiveProblem supplies uncertainty and, for trust-constr, the
optimizer. Other methods use only the objective and optional gradient.

## `bootstrap`

```python
bootstrap(statistic: 'Callable[..., ArrayLike | Estimate]', data: 'tuple[ArrayLike, ...]', *, rng: 'np.random.Generator', level: 'float' = 0.95, method: 'str' = 'BCa', n_resamples: 'int' = 999, batch: 'int' = 256, names: 'Sequence[str] | None' = None) -> 'BootstrapResult'
```

Bootstrap independent rows, preserving pairing across a tuple of arrays.

statistic(*data) returns a scalar, vector or converged Estimate. Each array
has n rows; trailing dimensions are preserved. No failed fit is discarded.
RNG must be a numpy.random.Generator. This is not a time-series bootstrap.

## `delta_method`

```python
delta_method(estimate: 'Estimate', function: 'Callable[[FloatArray], ArrayLike]', *, jacobian: 'Callable[[FloatArray], ArrayLike] | None' = None, names: 'Sequence[str] | None' = None) -> 'Estimate'
```

Propagate covariance as J C J.T for a smooth scalar/vector transformation.

Nonlinear transformations use asymptotic normal inference, not exact t
inference. A Laplace posterior covariance retains its credible meaning.

## `elastic_net`

```python
elastic_net(design: 'ArrayLike', response: 'ArrayLike', *, alpha: 'float' = 1.0, l1_ratio: 'float' = 0.5, names: 'Sequence[str] | None' = None, tolerance: 'float' = 1e-08, maxiter: 'int' = 10000) -> 'Estimate'
```

Fit y=X@theta with an elastic-net penalty and no implicit intercept/scaling.

Design is (n,p), response (n,); alpha>0 and l1_ratio in [0,1].
Return Estimate; penalized curvature is not unpenalized sampling covariance.

## `elastic_net_penalty`

```python
elastic_net_penalty(alpha: 'float' = 1.0, l1_ratio: 'float' = 0.5, *, center: 'ArrayLike' = 0.0, weights: 'ArrayLike' = 1.0) -> 'ElasticNetPenalty'
```

Construct alpha times a weighted mixture of L1 and half-squared L2 distances.

Center and nonnegative weights broadcast to parameter coordinates; l1_ratio
sets the L1 share. The returned callable is a penalty, not a normalized prior.

## `gaussian_prior`

```python
gaussian_prior(loc: 'ArrayLike' = 0.0, scale: 'ArrayLike' = 1.0) -> 'IndependentPrior'
```

Independent Gaussian log prior; scale is a standard deviation.

## `l1_penalty`

```python
l1_penalty(alpha: 'float' = 1.0, *, center: 'ArrayLike' = 0.0, weights: 'ArrayLike' = 1.0) -> 'ElasticNetPenalty'
```

Construct alpha*sum(weights*abs(theta-center)); weights broadcast to coordinates.

## `l2_penalty`

```python
l2_penalty(alpha: 'float' = 1.0, *, center: 'ArrayLike' = 0.0, weights: 'ArrayLike' = 1.0) -> 'ElasticNetPenalty'
```

Construct alpha/2*sum(weights*(theta-center)**2); weights broadcast to coordinates.

## `laplace_interval`

```python
laplace_interval(estimate: 'Estimate', *, level: 'float' = 0.95) -> 'Interval'
```

Marginal approximate credible intervals from local posterior curvature.

## `laplace_prior`

```python
laplace_prior(loc: 'ArrayLike' = 0.0, scale: 'ArrayLike' = 1.0) -> 'IndependentPrior'
```

Independent Laplace log prior (SD=sqrt(2)*scale); nonsmooth at loc.

MAP at a cusp does not have a Laplace curvature approximation: pass
uncertainty=False and choose a suitable nonsmooth solver explicitly.

## `lasso`

```python
lasso(design: 'ArrayLike', response: 'ArrayLike', *, alpha: 'float' = 1.0, names: 'Sequence[str] | None' = None, tolerance: 'float' = 1e-08, maxiter: 'int' = 10000) -> 'Estimate'
```

Fit y=X@theta with an L1 penalty and no implicit intercept/scaling.

Design is (n,p), response (n,); alpha>0. Return a regularized Estimate.

## `least_squares`

```python
least_squares(residuals: 'Callable[[FloatArray], ArrayLike]', initial: 'ParameterSpace | ArrayLike', *, jacobian: 'Callable[[FloatArray], ArrayLike] | None' = None, loss: 'str' = 'linear', f_scale: 'float' = 1.0, absolute_sigma: 'bool' = False, estimator: 'Estimator[LeastSquaresProblem] | None' = None, check_convergence: 'bool' = True) -> 'Estimate'
```

Fit r(theta). Divide residuals by known SDs and set absolute_sigma=True.

Otherwise covariance is scaled by SSE/(n-p). Robust losses return point
estimates without attaching ordinary least-squares covariance.

## `likelihood`

```python
likelihood(factory: 'Callable[[FloatArray], LogDensityLaw]', data: 'ArrayLike', *, frequency_weights: 'ArrayLike | None' = None) -> 'Likelihood'
```

Compose a theta -> Distribution factory with scalar or vector observations.

Any object with log_density works; the probability family needs no changes.
For conditional models use Likelihood(data, log_density) directly.

## `linear_least_squares`

```python
linear_least_squares(design: 'ArrayLike', response: 'ArrayLike', *, names: 'Sequence[str] | None' = None, weights: 'ArrayLike | None' = None, sigma: 'ArrayLike | None' = None, covariance_type: 'str' = 'nonrobust') -> 'Estimate'
```

OLS, WLS (precision weights) or GLS (relative observation covariance).

## `maximum_a_posteriori`

```python
maximum_a_posteriori(model: 'Likelihood', log_prior: 'Callable[[FloatArray], float]', initial: 'ParameterSpace | ArrayLike', *, gradient: 'Callable[[FloatArray], ArrayLike] | None' = None, hessian: 'Callable[[FloatArray], ArrayLike] | None' = None, constraints: 'Sequence[LinearConstraint | NonlinearConstraint]' = (), uncertainty: 'bool' = True, estimator: 'Estimator[ObjectiveProblem] | None' = None, check_convergence: 'bool' = True) -> 'Estimate'
```

Minimize -log likelihood - log prior in the supplied parameter coordinates.

Derivatives are of the whole negative log posterior. uncertainty requests a
local Laplace posterior approximation, valid only at a smooth interior mode.

## `maximum_likelihood`

```python
maximum_likelihood(model: 'Likelihood', initial: 'ParameterSpace | ArrayLike', *, gradient: 'Callable[[FloatArray], ArrayLike] | None' = None, hessian: 'Callable[[FloatArray], ArrayLike] | None' = None, constraints: 'Sequence[LinearConstraint | NonlinearConstraint]' = (), uncertainty: 'bool' = True, estimator: 'Estimator[ObjectiveProblem] | None' = None, check_convergence: 'bool' = True) -> 'Estimate'
```

Minimize the summed negative log likelihood; derivatives use that sign.

uncertainty requests inverse observed information for an interior fit.

## `minimize_estimate`

```python
minimize_estimate(objective: 'Callable[[FloatArray], float]', initial: 'ParameterSpace | ArrayLike', *, gradient: 'Callable[[FloatArray], ArrayLike] | None' = None, hessian: 'Callable[[FloatArray], ArrayLike] | None' = None, constraints: 'Sequence[LinearConstraint | NonlinearConstraint]' = (), estimator: 'Estimator[ObjectiveProblem] | None' = None, check_convergence: 'bool' = True) -> 'Estimate'
```

Estimate parameters by minimizing a user criterion; no automatic covariance.

## `regularized_linear`

```python
regularized_linear(design: 'ArrayLike', response: 'ArrayLike', *, names: 'Sequence[str] | None' = None, alpha: 'float' = 1.0, l1_ratio: 'float' = 0.0, tolerance: 'float' = 1e-08, maxiter: 'int' = 10000, check_convergence: 'bool' = True) -> 'Estimate'
```

Fit a penalized linear design (n,p) against scalar response (n,).

No intercept or scaling is added. l1_ratio interpolates ridge and lasso.
Return Estimate; inspect diagnostics for convergence and uncertainty semantics.

## `ridge`

```python
ridge(design: 'ArrayLike', response: 'ArrayLike', *, alpha: 'float' = 1.0, names: 'Sequence[str] | None' = None) -> 'Estimate'
```

Fit y=X@theta with an L2 penalty and no implicit intercept/scaling.

Design is (n,p), response (n,); alpha>0. Return a regularized Estimate.

## `robust_linear`

```python
robust_linear(design: 'ArrayLike', response: 'ArrayLike', *, names: 'Sequence[str] | None' = None, loss: 'str' = 'huber', tuning: 'float | None' = None, covariance_type: 'str' = 'H1', tolerance: 'float' = 1e-08, maxiter: 'int' = 200, check_convergence: 'bool' = True) -> 'Estimate'
```

Fit a robust M-estimator to design (n,p) and response (n,).

No intercept or scaling is implicit. loss/tuning select the residual influence
function; covariance_type selects the asymptotic covariance estimator.

## `wald_interval`

```python
wald_interval(estimate: 'Estimate', *, level: 'float' = 0.95) -> 'Interval'
```

Marginal two-sided confidence intervals. No clipping to parameter bounds.

## `BootstrapResult`

```python
BootstrapResult(estimate: 'Estimate', interval: 'Interval', draws: 'FloatArray') -> None
```

Bootstrap parameter replicates and interval information; resampling assumptions determine inferential meaning.

## `ConvergenceError`

```python
ConvergenceError(result: 'Estimate') -> 'None'
```

Raised when a checked numerical method does not converge; result retains the attempted solution and diagnostics.

## `Diagnostics`

```python
Diagnostics(converged: 'bool' = True, message: 'str' = '', backend: 'str' = 'user', iterations: 'int | None' = None, evaluations: 'int | None' = None, rank: 'int | None' = None, constraint_violation: 'float' = 0.0, uncertainty_note: 'str' = '') -> None
```

Backend termination diagnostics. Inspect converged and message before using a fitted approximation; iterations may be unavailable.

## `Estimate`

```python
Estimate(parameters: 'ArrayLike', *, names: 'Sequence[str] | None' = None, diagnostics: 'Diagnostics | None' = None, objective: 'float | None' = None, covariance: 'ArrayLike | None' = None, covariance_kind: 'str | None' = None, df: 'float | None' = None, nobs: 'int | None' = None, residuals: 'ArrayLike | None' = None) -> 'None'
```

One parameter vector; missing uncertainty stays None.

Covariance is in the coordinates of parameters. df controls the Student-t
reference for frequentist intervals; None means an asymptotic normal law.
A Laplace covariance describes local posterior curvature, not coverage.

## `EstimationProblem`

Structural parameter-estimation problem; names defines parameter coordinate order.

### `EstimationProblem.validate_result`

```python
validate_result(self, result: 'Estimate') -> 'None'
```

## `Estimator`

A callable accepting one problem and returning a checked Estimate.

## `Interval`

```python
Interval(lower: 'ArrayLike', upper: 'ArrayLike', level: 'float', method: 'str', kind: 'str' = 'confidence', names: 'tuple[str, ...] | None' = None) -> 'None'
```

Lower and upper parameter bounds with an explicit probability level and interval method.

## `LeastSquaresProblem`

```python
LeastSquaresProblem(residuals: 'Callable[[FloatArray], ArrayLike]', space: 'ParameterSpace', jacobian: 'Callable[[FloatArray], ArrayLike] | None' = None) -> None
```

Residual vector r(theta), optionally already whitened by known noise.

The criterion is half the sum of squared residuals for linear loss. An
analytic Jacobian must differentiate the same (possibly whitened) residuals.

### `LeastSquaresProblem.derivative`

```python
derivative(self, theta: 'ArrayLike')
```

### `LeastSquaresProblem.evaluate`

```python
evaluate(self, theta: 'ArrayLike') -> 'FloatArray'
```

### `LeastSquaresProblem.validate_result`

```python
validate_result(self, result)
```

## `Likelihood`

```python
Likelihood(data: 'ArrayLike', log_density: 'Callable[[FloatArray, FloatArray], ArrayLike]', frequency_weights: 'ArrayLike | None' = None) -> 'None'
```

Independent observation contributions log_density(theta, data) -> (n,).

Rows are observations. Optional integer frequency weights mean literal
replication; they are not precision, survey or importance weights. Zero
frequency rows are removed before calling the model. Constants are retained.

### `Likelihood.contributions`

```python
contributions(self, theta: 'ArrayLike')
```

### `Likelihood.log_likelihood`

```python
log_likelihood(self, theta: 'ArrayLike') -> 'float'
```

### `Likelihood.negative_log_likelihood`

```python
negative_log_likelihood(self, theta: 'ArrayLike')
```

## `LinearConstraint`

```python
LinearConstraint(matrix: 'ArrayLike', lower: 'ArrayLike' = -inf, upper: 'ArrayLike' = inf) -> 'None'
```

lower <= matrix @ theta <= upper; equal endpoints impose equality.

### `LinearConstraint.evaluate`

```python
evaluate(self, theta: 'ArrayLike')
```

## `LinearProblem`

```python
LinearProblem(design: 'ArrayLike', response: 'ArrayLike', names: 'Sequence[str] | None' = None, weights: 'ArrayLike | None' = None, sigma: 'ArrayLike | None' = None) -> 'None'
```

y = X @ theta + error, with no implicit intercept or standardization.

weights are strictly positive relative precisions, or sigma is a positive
definite relative observation covariance. The two are mutually exclusive.

### `LinearProblem.validate_result`

```python
validate_result(self, result)
```

## `NonlinearConstraint`

```python
NonlinearConstraint(function: 'Callable[[FloatArray], ArrayLike]', lower: 'ArrayLike', upper: 'ArrayLike', jacobian: 'Callable[[FloatArray], ArrayLike] | None' = None) -> 'None'
```

lower <= function(theta) <= upper; function returns shape (m,).

### `NonlinearConstraint.derivative`

```python
derivative(self, theta: 'ArrayLike')
```

### `NonlinearConstraint.evaluate`

```python
evaluate(self, theta: 'ArrayLike')
```

## `ObjectiveProblem`

```python
ObjectiveProblem(objective: 'Callable[[FloatArray], float]', space: 'ParameterSpace', gradient: 'Callable[[FloatArray], ArrayLike] | None' = None, hessian: 'Callable[[FloatArray], ArrayLike] | None' = None, constraints: 'Sequence[LinearConstraint | NonlinearConstraint]' = (), curvature: 'str | None' = None, nobs: 'int | None' = None) -> None
```

Minimize a scalar criterion. curvature labels its inverse Hessian meaning.

+inf is a permitted trial objective outside likelihood/prior support; NaN
and -inf are rejected. Derivatives are of this entire minimized objective.

### `ObjectiveProblem.derivative`

```python
derivative(self, theta: 'ArrayLike')
```

### `ObjectiveProblem.evaluate`

```python
evaluate(self, theta: 'ArrayLike')
```

### `ObjectiveProblem.second_derivative`

```python
second_derivative(self, theta: 'ArrayLike')
```

### `ObjectiveProblem.validate_result`

```python
validate_result(self, result)
```

### `ObjectiveProblem.violation`

```python
violation(self, theta: 'ArrayLike')
```

## `ParameterSpace`

```python
ParameterSpace(initial: 'ArrayLike', names: 'Sequence[str] | None' = None, lower: 'ArrayLike' = -inf, upper: 'ArrayLike' = inf) -> 'None'
```

A finite initial vector and closed box bounds (scalar or shape (p,)).

Equal bounds are excluded: fix a parameter in the model callback instead.
Positivity uses an explicit positive lower bound or log coordinates.

### `ParameterSpace.at_boundary`

```python
at_boundary(self, value, tolerance: 'float' = 1e-06) -> 'bool'
```

### `ParameterSpace.point`

```python
point(self, value: 'ArrayLike')
```

### `ParameterSpace.violation`

```python
violation(self, value) -> 'float'
```

## `fit`

```python
fit(problem: 'P', estimator: 'Estimator[P]', *, check_convergence: 'bool' = True) -> 'Estimate'
```

Estimate a checked problem using a compatible built-in or user Estimator.

Return Estimate with named parameter coordinates, objective diagnostics and
optional uncertainty. check_convergence=True raises ConvergenceError on failure.

## `LogDensityLaw`

Scalar/vector observation law; one log density per observation row.

### `LogDensityLaw.log_density`

```python
log_density(self, values: 'ArrayLike', /) -> 'ArrayLike'
```
