# Statistical estimation

Family 3 turns supplied observations, likelihoods and residual functions into
parameter estimates. It is independent of data acquisition and applications.
Mature libraries own the optimization, fitting and resampling algorithms.

## Finite acceptance catalog

| Mathematical task | Implementation |
|---|---|
| User-defined scalar estimation objectives | SciPy minimize, with gradients, named parameters, box bounds and linear/nonlinear equality/inequality constraints |
| Likelihood and maximum likelihood | Checked per-observation log likelihoods, distribution-family composition, frequency weights, observed-information uncertainty |
| MAP and penalties | User-supplied log prior; Gaussian/Laplace priors, L1/L2/elastic-net penalties; explicitly labelled Laplace posterior approximation |
| Nonlinear least squares | SciPy least_squares; residual Jacobians, bounds, known/estimated noise scaling and robust losses |
| Linear coefficient estimation | statsmodels OLS, WLS and GLS; classical and HC0–HC3 covariance |
| Robust linear coefficient estimation | statsmodels RLM with Huber or Tukey losses and H1–H3 covariance |
| Regularized linear coefficient estimation | scikit-learn ridge, lasso and elastic net under one documented objective normalization |
| Parameter uncertainty | Normal/Student-t Wald intervals, Laplace credible intervals, delta method, paired-row percentile/basic/BCa bootstrap |
| Extensibility and verification | Public problem/tool/result contracts, analytic identities, generated-data recovery, a script and an executable notebook |

The catalog excludes general predictive learning (family 6), posterior sampling
(family 11), and general mathematical optimization (family 29). Local optimizers
do not certify a global optimum. Arbitrary constrained or penalized estimates do
not automatically acquire frequentist covariance matrices.

## Start with a mathematical model

```python
import numpy as np
from convexity.math import distributions as dist, estimation as est

rng = np.random.default_rng(42)
y = dist.normal(2, 1.5).sample(500, rng=rng)
model = est.likelihood(lambda theta: dist.normal(theta[0], np.exp(theta[1])), y)
space = est.ParameterSpace([0., 0.], names=("mu", "log_sigma"))
result = est.maximum_likelihood(model, space)
result.named_parameters
result.standard_errors
est.wald_interval(result)
sigma = est.delta_method(result, lambda theta: np.exp(theta[1]), names=("sigma",))
```

The Gaussian MLE is the sample mean and sample standard deviation with `ddof=0`.
Log scale is a deliberate choice of coordinates; there are no implicit parameter
transforms. `likelihood` accepts any factory returning an object with
`log_density`, including compatible laws in family 2.

Run the [problem-book script](examples/statistical_estimation.py) or open
the [worked notebook](notebooks/statistical_estimation.ipynb). Both use
supplied or generated mathematical inputs.

## Three layers

- `_backends/`: SciPy optimization, covariance solves, reference quantiles and
  bootstrap; statsmodels linear/robust fits; scikit-learn penalized fits.
- `interfaces/`: `ParameterSpace`, constraints, `Likelihood`, `ObjectiveProblem`,
  `LeastSquaresProblem`, `LinearProblem`, `Estimator`, `Estimate`, `Diagnostics`,
  `Interval`, `BootstrapResult`, and `fit`.
- `builtins/`: mathematical functions and configurable callable tools. All public
  names are also exported from `convexity.math.estimation`.

The completed numerical family supplies finite-difference Hessians and Jacobians
when analytical derivatives are absent, loading only when needed. statsmodels
and scikit-learn are declared dependencies but load only on use. Importing
estimation loads no application, probability, trading or tensor stack. Neither
existing mathematical family is modified.

## Shapes, ownership and failures

| Input or result | Convention |
|---|---|
| Parameters and initial guesses | Finite real `(p,)`, including scalar parameters |
| Parameter names | Unique nonempty names in coordinate order; defaults `theta_0`, … |
| Observations | At least one observation axis; scalar data `(n,)`, vector data `(n,d)` |
| Likelihood callback | `log_density(theta, data)` returns `(n,)`, one joint contribution per row |
| Objective derivatives | Scalar objective; gradient `(p,)`; symmetric Hessian `(p,p)` |
| Residuals and Jacobian | `(n,)` and `(n,p)`, with identical residual scaling/order |
| Linear model | Design `(n,p)`, response `(n,)`; explicitly supply every column |
| Estimate | Parameters `(p,)`, covariance `(p,p)` or `None`, optional residuals `(n,)` |
| Intervals | Lower/upper `(p,)`; marginal, two-sided, not simultaneous regions |
| Bootstrap draws | `(n_resamples,p)`, including `(n_resamples,1)` for scalar statistics |

Constructors copy inputs into read-only float64 arrays. Numeric strings,
booleans, complex values, empty observations, NaNs and infinite data are rejected.
Callbacks receive read-only inputs and must be deterministic, shape-preserving
and nonmutating. Infinite trial objectives and negative-infinite log densities
can express zero support. Initial and successful final objectives must be finite.
The callback must be defined wherever the chosen solver evaluates it; declare
domain restrictions as bounds or change coordinates explicitly.

`fit` checks result type, names, dimensions, feasibility and convergence.
`ConvergenceError` retains the failed estimate as `.result`.
`check_convergence=False` returns failures for inspection; intervals still reject
them. Diagnostics retain backend identity, available iteration/evaluation counts,
rank, constraint violation and the reason covariance is absent. A converged point
fit can have no covariance. Pseudoinverses are not presented as identified
parameter uncertainty.

## Likelihood, constraints and MAP

`Likelihood(data, log_density, frequency_weights=...)` supports custom conditional
models. Contributions are summed, not averaged. Nonnegative integer frequency
weights mean repeated independent observations, and `nobs` is their sum.
Zero-frequency rows are removed before the callback. Precision, importance and
survey weights have different meanings and are not accepted here. The model owns
the independence and reference-measure assumptions.

```python
# Conditional Gaussian likelihood for y_i = beta*x_i + error_i.
xy = np.array([[1., 2.], [2., 4.], [3., 3.]])
model = est.Likelihood(xy, lambda t, rows:
    dist.normal(0, 1).log_density(rows[:, 1] - t[0] * rows[:, 0]))
result = est.maximum_likelihood(model, [0.])

# Two nonnegative coefficients whose sum is one.
space = est.ParameterSpace([.5, .5], lower=0., upper=1.)
constraint = est.LinearConstraint([[1., 1.]], lower=1., upper=1.)
result = est.minimize_estimate(lambda t: np.sum((t - [.8, .7])**2),
    space, constraints=[constraint])  # [.55, .45]
```

`ParameterSpace` supports scalar/vector closed bounds with `lower < upper`.
Fix parameters inside the callback rather than using equal box endpoints.
`LinearConstraint(A, lower, upper)` means `lower <= A @ theta <= upper`.
`NonlinearConstraint(function, lower, upper, jacobian=None)` uses a vector-valued
function and the same convention. Equal constraint endpoints impose equality;
one-sided infinities express inequalities. Feasibility uses absolute tolerance
`1e-6`, so scale constraints sensibly. Initial guesses must satisfy box bounds but
may violate general constraints.

`ScipyMinimizer` selects SLSQP for bounds/general constraints and L-BFGS-B for
unbounded objectives. Explicit choices include BFGS, Powell and trust-constr.
Unsupported combinations raise instead of dropping constraints or gradients.
SciPy supplies three-point gradients when needed. An analytic Hessian is used by
trust-constr and for uncertainty; other methods do not use it to optimize.
Explicit L-BFGS-B fits guard against false convergence after infinite trial
values. These are local solvers, without a global optimality certificate.

`maximum_likelihood` minimizes negative **summed** log likelihood. Its gradient
and Hessian callbacks differentiate that entire minimized objective, including
the sign. `maximum_a_posteriori(model, log_prior, initial)` minimizes negative log
likelihood minus log prior; its derivatives include both terms.

`gaussian_prior(loc, scale)` and `laplace_prior(loc, scale)` return independent
coordinate log-prior callables with scalar/vector parameters. Gaussian scale is
SD; Laplace scale is SD divided by `sqrt(2)`. MAP depends on coordinates: a prior
transformed between coordinates requires the corresponding density Jacobian.
The nonsmooth built-in Laplace prior requires `uncertainty=False` and a suitable
solver, such as explicit Powell for small problems. Users own the smoothness and
mathematical validity of arbitrary priors and penalties.

## Least squares, robust estimation and regularization

```python
x = np.linspace(0, 4, 80)
y = 3 * np.exp(-.7 * x)
result = est.least_squares(lambda t: t[0] * np.exp(-t[1] * x) - y,
    est.ParameterSpace([2., .5], lower=0.))

X = np.column_stack([np.ones(len(x)), x])
ols = est.linear_least_squares(X, y, names=("intercept", "slope"))
wls = est.linear_least_squares(X, y, weights=np.linspace(1, 2, len(x)))
hc3 = est.linear_least_squares(X, y, covariance_type="HC3")
robust = est.robust_linear(X, y, loss="huber")
ridge = est.ridge(X, y, alpha=.1)
```

Nonlinear `least_squares` minimizes `sum(r**2)/2` for linear loss. `LeastSquares`
configures TRF, dogbox or LM; LM requires unbounded parameters and linear loss.
For independent known measurement SDs, standardize residuals and set
`absolute_sigma=True`: covariance is `(J.T @ J)^-1`. With unknown common residual
scale, the default multiplies by `SSE/(n-p)`. Nonlinear inference is a local
linearization. Known general noise covariance can be handled by explicitly
whitening residuals and their Jacobian together.

Robust losses `soft_l1`, `huber`, `cauchy` and `arctan` use SciPy's implementations
with positive `f_scale` in residual units. They return no ordinary least-squares
covariance. With a custom `estimator`, configure options on that tool; conflicting
convenience options are rejected.

Linear least squares uses statsmodels OLS, WLS or GLS. WLS weights are positive
relative inverse error variances. GLS `sigma` is a positive-definite relative
observation covariance. Both estimate a common scale from residuals, and weights
and sigma are mutually exclusive. Classical covariance uses a Student-t
reference with `n-rank(X)` degrees of freedom; HC0–HC3 use asymptotic normal
intervals. No intercept or scaling is inserted. Residuals have original response
units; objective is half the whitened SSE.

`robust_linear` uses statsmodels RLM with Huber (default tuning `1.345`) or Tukey
(`4.685`) loss, MAD scale estimation, and H1/H2/H3 covariance. Diagnostics check
coefficient convergence in the IRLS history. Robust covariance is asymptotic and
is absent at zero residual scale or rank deficiency. The joint scale estimate
does not yield a directly comparable SSE/likelihood objective, so `objective` is
`None`. Redescending Tukey loss can have local solutions.

All regularized linear constructors minimize

```text
||y - X theta||² / (2n)
    + alpha * [l1_ratio*||theta||_1 + (1-l1_ratio)*||theta||²/2].
```

`ridge`, `lasso` and `elastic_net` use ratios 0, 1 and `.5` (configurable).
`alpha` must be positive; use OLS for an unpenalized fit. The adapter converts
this normalization to scikit-learn's native ridge convention. Every explicit
column, including a constant, is penalized. No intercept is automatically
exempted. These biased, potentially nonsmooth estimates have no automatic
covariance. Backend convergence warnings become failed fit diagnostics.

`l1_penalty`, `l2_penalty` and `elastic_net_penalty` supply callable penalties for
custom scalar criteria. Their formula is
`alpha*sum(weights*(ratio*abs(theta-center)+(1-ratio)*(theta-center)**2/2))`.
Zero weights exempt selected coordinates. Add the penalty explicitly; its scale
is never implicitly divided by observation count.

## Interpreting uncertainty

- MLE covariance is inverse observed information: the Hessian of the negative
  **sum** log likelihood. It assumes a smooth, identifiable, regular interior
  model with appropriate likelihood specification. It is not a misspecification
  sandwich estimate.
- MAP covariance is inverse negative-log-posterior curvature, labelled `laplace`.
  `laplace_interval` supplies approximate credible intervals. `wald_interval`
  rejects posterior covariance.
- Use analytic Hessians when possible. Otherwise family 1's checked finite
  differences use steps kept inside box bounds. Nonconvergence or nonpositive,
  asymmetric or rank-deficient information gives no covariance and a diagnostic.
  General constraints and active bounds disable this approximation. Optimizer
  `hess_inv` approximations are never substituted for observed information.
- `delta_method(estimate, function, jacobian=...)` propagates covariance as
  `J C J.T`. Jacobians are `(output,input)`, including `(1,p)` for scalar outputs.
  Family 1 differentiates when no Jacobian is supplied. Inference is a first-order
  normal approximation; posterior covariance retains its credible interpretation.
- Intervals are marginal and never clipped to bounds. Zero residual scale can
  yield zero-width classical intervals, which do not establish a realistic noise
  model. Covariance validation allows small floating-point roundoff in symmetry
  and positive semidefiniteness.

```python
resampled = est.bootstrap(
    lambda design, response: est.linear_least_squares(design, response),
    (X, y), rng=np.random.default_rng(8), method="BCa", n_resamples=999)
resampled.interval
resampled.estimate.covariance
resampled.draws
```

Bootstrap accepts a **tuple** of equal-length arrays and a callable returning a
scalar, vector or converged `Estimate`. It preserves whole rows and pairing,
including vector observations and design/response pairs. SciPy owns resampling,
percentile/basic/BCa intervals and the BCa jackknife. Supply a
`numpy.random.Generator`; global RNG state is never used. Draws and the original
point estimate are retained; covariance uses `ddof=1` across estimates. Even when
the statistic is a MAP estimate, these draws describe sampling variability,
not a posterior law.

The row bootstrap assumes independent identically distributed rows; it is not a
time-series/block or fixed-design residual bootstrap. Failed fits and changed
statistic shapes abort without discarding draws. Degenerate BCa intervals raise
without silently selecting a different method.

## Supply your own estimator

```python
problem = est.ObjectiveProblem(lambda theta: (theta[0] - 3)**2,
    est.ParameterSpace([0.], names=("location",)))

def analytic_estimator(problem):
    return est.Estimate([3.], names=problem.space.names, objective=0.,
        diagnostics=est.Diagnostics(backend="analytic quadratic"))

result = est.fit(problem, analytic_estimator)
```

No subclass or registry is needed. Tools own their algorithms, convergence claims
and covariance interpretation. The problem validates the returned contract.
A custom problem can implement `validate_result` to use the same `fit` function.
Likelihoods, residuals, constraints, priors and bootstrap statistics are also
extensible without changes to Convexity internals.

## Verification and references

```bash
python -m pytest -q tests/math/estimation
python -m pyright --pythonpath "$(command -v python)"
python docs/math/examples/statistical_estimation.py
```

Tests verify Gaussian/discrete MLE identities, conjugate MAP, information matrices,
nonlinear recovery, OLS/WLS/GLS whitening, HC sandwich identities, orthogonal
regularization closed forms, outlier resistance, constraints, failures, bootstrap
agreement and pairing, uncertainty transforms, invalid inputs and custom tools.
The family suite also runs against NumPy 1.26.4, SciPy 1.15.3,
statsmodels 0.14.4 and scikit-learn 1.6.1 to check the declared dependency floors.

Backend contracts: [SciPy minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html),
[least squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html),
[bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html),
[statsmodels OLS](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html),
[robust covariance](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_robustcov_results.html),
[RLM fitting](https://www.statsmodels.org/stable/generated/statsmodels.robust.robust_linear_model.RLM.fit.html),
[scikit-learn elastic net](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html).
