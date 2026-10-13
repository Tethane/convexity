# `convexity.math.dependence`

[Reference index](README.md) · [Contract conventions](../public_contracts.md)

Covariance, dependence and tails on supplied mathematical inputs.

    from convexity.math import dependence as dep
    covariance = dep.LedoitWolf().fit(training_matrix)
    dependence = dep.fit_copula(training_matrix, family="student_t", df=5)

Backends load on use. No data, calendar, forecasting or trading imports.

## `ClaytonCopula`

```python
ClaytonCopula(theta: 'float') -> 'None'
```

Archimedean Clayton dependence for unit-interval margins; parameter restrictions depend on dimension.

### `ClaytonCopula.cdf`

```python
cdf(self, uniforms: 'ArrayLike', *, rng: 'np.random.Generator | None' = None)
```

Elliptical adapters require rng for randomized SciPy integration.

### `ClaytonCopula.density`

```python
density(self, uniforms) -> 'FloatArray'
```

### `ClaytonCopula.log_density`

```python
log_density(self, uniforms: 'ArrayLike')
```

### `ClaytonCopula.sample`

```python
sample(self, size: 'SampleSize' = (), *, rng: 'np.random.Generator')
```

### `ClaytonCopula.tail_dependence`

```python
tail_dependence(self, i: 'int' = 0, j: 'int' = 1)
```

Pairwise (lower, upper) limiting conditional exceedance probabilities.

## `EWMACovariance`

```python
EWMACovariance(decay: 'float' = 0.94, ddof: 'int' = 0, assume_centered: 'bool' = False) -> None
```

Exponentially weighted covariance specification with explicit decay and centering settings.

### `EWMACovariance.fit`

```python
fit(self, samples: 'ArrayLike') -> 'CovarianceEstimate'
```

## `GaussianCopula`

```python
GaussianCopula(correlation: 'ArrayLike') -> 'None'
```

Gaussian dependence parameterized by a correlation matrix; marginal distributions are supplied separately.

### `GaussianCopula.cdf`

```python
cdf(self, uniforms: 'ArrayLike', *, rng: 'np.random.Generator | None' = None)
```

Elliptical adapters require rng for randomized SciPy integration.

### `GaussianCopula.density`

```python
density(self, uniforms) -> 'FloatArray'
```

### `GaussianCopula.log_density`

```python
log_density(self, uniforms: 'ArrayLike')
```

### `GaussianCopula.sample`

```python
sample(self, size: 'SampleSize' = (), *, rng: 'np.random.Generator')
```

### `GaussianCopula.tail_dependence`

```python
tail_dependence(self, i: 'int' = 0, j: 'int' = 1)
```

Pairwise (lower, upper) limiting conditional exceedance probabilities.

## `GeneralizedExtremeValue`

```python
GeneralizedExtremeValue(maxiter: 'int' = 3000, tolerance: 'float' = 1e-08, check_convergence: 'bool' = True) -> None
```

Fit a generalized extreme-value law to block maxima, with checked numerical convergence.

### `GeneralizedExtremeValue.fit`

```python
fit(self, maxima: 'ArrayLike') -> 'TailFit'
```

## `GeneralizedPareto`

```python
GeneralizedPareto(maxiter: 'int' = 3000, tolerance: 'float' = 1e-08, check_convergence: 'bool' = True) -> None
```

Fit a generalized Pareto law to nonnegative threshold excesses, with checked numerical convergence.

### `GeneralizedPareto.fit`

```python
fit(self, excesses: 'ArrayLike') -> 'TailFit'
```

## `GumbelCopula`

```python
GumbelCopula(theta: 'float') -> 'None'
```

Archimedean Gumbel dependence for unit-interval margins; theta controls upper-tail dependence.

### `GumbelCopula.cdf`

```python
cdf(self, uniforms: 'ArrayLike', *, rng: 'np.random.Generator | None' = None)
```

Elliptical adapters require rng for randomized SciPy integration.

### `GumbelCopula.density`

```python
density(self, uniforms) -> 'FloatArray'
```

### `GumbelCopula.log_density`

```python
log_density(self, uniforms: 'ArrayLike')
```

### `GumbelCopula.sample`

```python
sample(self, size: 'SampleSize' = (), *, rng: 'np.random.Generator')
```

### `GumbelCopula.tail_dependence`

```python
tail_dependence(self, i: 'int' = 0, j: 'int' = 1)
```

Pairwise (lower, upper) limiting conditional exceedance probabilities.

## `IndependenceCopula`

```python
IndependenceCopula(dimension: 'int' = 2) -> 'None'
```

Product copula of the declared dimension; supplies no cross-coordinate dependence.

### `IndependenceCopula.cdf`

```python
cdf(self, uniforms: 'ArrayLike', *, rng: 'np.random.Generator | None' = None)
```

Elliptical adapters require rng for randomized SciPy integration.

### `IndependenceCopula.density`

```python
density(self, uniforms) -> 'FloatArray'
```

### `IndependenceCopula.log_density`

```python
log_density(self, uniforms: 'ArrayLike')
```

### `IndependenceCopula.sample`

```python
sample(self, size: 'SampleSize' = (), *, rng: 'np.random.Generator')
```

### `IndependenceCopula.tail_dependence`

```python
tail_dependence(self, i: 'int' = 0, j: 'int' = 1)
```

Pairwise (lower, upper) limiting conditional exceedance probabilities.

## `KendallCopula`

```python
KendallCopula(family: 'str' = 'gaussian', df: 'float | None' = None, repair_correlation: 'bool' = False) -> None
```

Estimate copula dependence from Kendall rank correlations, with explicit family and repair choices.

### `KendallCopula.fit`

```python
fit(self, samples: 'ArrayLike') -> 'CopulaFit'
```

## `LedoitWolf`

```python
LedoitWolf(assume_centered: 'bool' = False) -> None
```

Estimate covariance using data-driven shrinkage toward a scaled identity target.

### `LedoitWolf.fit`

```python
fit(self, samples: 'ArrayLike') -> 'CovarianceEstimate'
```

## `OAS`

```python
OAS(assume_centered: 'bool' = False) -> None
```

Estimate covariance using oracle-approximating shrinkage toward a scaled identity target.

### `OAS.fit`

```python
fit(self, samples: 'ArrayLike') -> 'CovarianceEstimate'
```

## `PositiveSemidefinite`

```python
PositiveSemidefinite(min_eigenvalue: 'float' = 0.0, tolerance: 'float' = 1e-10) -> None
```

Eigenvalue lower bound in matrix units; numerical membership tolerance.

### `PositiveSemidefinite.contains`

```python
contains(self, matrix: 'ArrayLike') -> 'bool'
```

### `PositiveSemidefinite.project`

```python
project(self, matrix: 'ArrayLike')
```

Frobenius projection by spectral clipping; diagonal may change.

## `SampleCovariance`

```python
SampleCovariance(ddof: 'int' = 1, assume_centered: 'bool' = False) -> None
```

Empirical covariance from observation rows; ddof and assumed centering are explicit.

### `SampleCovariance.fit`

```python
fit(self, samples: 'ArrayLike') -> 'CovarianceEstimate'
```

## `ShrunkCovariance`

```python
ShrunkCovariance(shrinkage: 'float' = 0.1, assume_centered: 'bool' = False) -> None
```

Blend empirical covariance with a scaled identity target using an explicit shrinkage strength.

### `ShrunkCovariance.fit`

```python
fit(self, samples: 'ArrayLike') -> 'CovarianceEstimate'
```

## `StudentT`

```python
StudentT(df: 'float' = 5.0, estimate_df: 'bool' = False, df_bounds: 'tuple[float, float]' = (0.5, 200.0), maxiter: 'int' = 1000, tolerance: 'float' = 1e-10, check_convergence: 'bool' = True) -> None
```

Elliptical Student-t likelihood fit, optionally estimating df within bounds.

### `StudentT.fit`

```python
fit(self, samples: 'ArrayLike') -> 'StudentTResult'
```

## `StudentTCopula`

```python
StudentTCopula(correlation: 'ArrayLike', df: 'float') -> 'None'
```

Student-t dependence with a correlation matrix and degrees of freedom; margins remain separate.

### `StudentTCopula.cdf`

```python
cdf(self, uniforms: 'ArrayLike', *, rng: 'np.random.Generator | None' = None)
```

Elliptical adapters require rng for randomized SciPy integration.

### `StudentTCopula.density`

```python
density(self, uniforms) -> 'FloatArray'
```

### `StudentTCopula.log_density`

```python
log_density(self, uniforms: 'ArrayLike')
```

### `StudentTCopula.sample`

```python
sample(self, size: 'SampleSize' = (), *, rng: 'np.random.Generator')
```

### `StudentTCopula.tail_dependence`

```python
tail_dependence(self, i: 'int' = 0, j: 'int' = 1)
```

Pairwise (lower, upper) limiting conditional exceedance probabilities.

## `block_maxima`

```python
block_maxima(samples: 'ArrayLike', block_size: 'int', *, drop_incomplete: 'bool' = False) -> 'FloatArray'
```

Nonoverlapping positional blocks. Discard a trailing partial block only explicitly.

## `correlation`

```python
correlation(samples: 'ArrayLike', *, method: 'str' = 'pearson') -> 'FloatArray'
```

Pearson, Spearman or Kendall tau-b on complete observation rows.

## `correlation_to_covariance`

```python
correlation_to_covariance(correlation: 'ArrayLike', standard_deviation: 'ArrayLike') -> 'FloatArray'
```

Return diag(std) @ correlation @ diag(std) with aligned coordinate order.

Standard deviations are nonnegative and carry the marginal measurement units.

## `covariance_to_correlation`

```python
covariance_to_correlation(covariance: 'ArrayLike') -> 'FloatArray'
```

Normalize a covariance matrix by marginal standard deviations.

Zero-variance coordinates have no defined correlation and are rejected.

## `ewma_covariance`

```python
ewma_covariance(samples: 'ArrayLike', *, decay: 'float' = 0.94, ddof: 'int' = 0, assume_centered: 'bool' = False) -> 'CovarianceEstimate'
```

Fit exponentially weighted covariance to time-ordered sample rows.

Decay in (0,1) discounts older rows; ddof controls the weight-based correction.
Return CovarianceEstimate in squared feature units.

## `fit_copula`

```python
fit_copula(samples: 'ArrayLike', *, family: 'str' = 'gaussian', df: 'float | None' = None, repair_correlation: 'bool' = False) -> 'CopulaFit'
```

Fit a copula by rank dependence from samples (sample,coordinate).

family selects the bounded supported catalog. Correlation repair is explicit;
Student-t degrees of freedom are supplied separately from marginal laws.

## `fit_factor_covariance`

```python
fit_factor_covariance(samples: 'ArrayLike', factors: 'ArrayLike', *, ddof: 'int' = 1, residual: 'str' = 'diagonal') -> 'FactorCovariance'
```

OLS exposures to supplied factors; factor/noise moments use n-ddof.

Full residual covariance reproduces sample covariance. Diagonal residuals
impose a model assumption. Neither estimates regression standard errors.

## `fit_gev`

```python
fit_gev(maxima: 'ArrayLike', *, maxiter: 'int' = 3000, tolerance: 'float' = 1e-08, check_convergence: 'bool' = True) -> 'TailFit'
```

Fit a generalized extreme-value law to scalar block maxima.

Blocks and their common horizon are supplied by the caller. Return a tail-fit
result with convergence diagnostics; this does not establish block independence.

## `fit_gpd`

```python
fit_gpd(excesses: 'ArrayLike', *, maxiter: 'int' = 3000, tolerance: 'float' = 1e-08, check_convergence: 'bool' = True) -> 'TailFit'
```

Fit a generalized Pareto law to nonnegative threshold excesses.

Inputs are excess amounts, not raw observations. Return a tail fit with explicit
shape/scale semantics and convergence diagnostics.

## `fit_student_t`

```python
fit_student_t(samples: 'ArrayLike', *, df: 'float' = 5.0, estimate_df: 'bool' = False, df_bounds: 'tuple[float, float]' = (0.5, 200.0), maxiter: 'int' = 1000, tolerance: 'float' = 1e-10, check_convergence: 'bool' = True) -> 'StudentTResult'
```

Fit a multivariate Student-t model to sample rows.

Scale is the t dispersion matrix, not covariance; covariance exists only for
df>2. Optional df estimation respects df_bounds and reports convergence.

## `generalized_extreme_value`

```python
generalized_extreme_value(xi: 'float', *, loc: 'float' = 0.0, scale: 'float' = 1.0) -> 'Distribution'
```

GEV block-maximum law; conventional xi is the negative of SciPy's c.

## `generalized_pareto`

```python
generalized_pareto(xi: 'float', *, scale: 'float' = 1.0, threshold: 'float' = 0.0) -> 'Distribution'
```

Law of threshold + GPD excess; xi>0 is heavy-tailed, xi<0 bounded.

## `multivariate_student_t`

```python
multivariate_student_t(df: 'float', loc: 'ArrayLike', shape: 'ArrayLike') -> 'Distribution'
```

Reuse the completed family-2 law, including moments and conditioning.

## `nearest_correlation`

```python
nearest_correlation(matrix: 'ArrayLike', *, min_eigenvalue: 'float' = 1e-10, maxiter: 'int' = 1000, check_convergence: 'bool' = True) -> 'MatrixRepair'
```

statsmodels alternating projection with fixed unit diagonal.

## `peaks_over_threshold`

```python
peaks_over_threshold(samples: 'ArrayLike', threshold: 'float', *, maxiter: 'int' = 3000, tolerance: 'float' = 1e-08, check_convergence: 'bool' = True) -> 'PeaksOverThresholdResult'
```

Fit the exceedance law above threshold from raw scalar samples.

Return threshold, exceedance frequency and GPD tail fit. Tail probabilities
combine observed exceedance mass with the conditional excess law.

## `pseudo_observations`

```python
pseudo_observations(samples: 'ArrayLike') -> 'FloatArray'
```

Columnwise average ranks divided by n+1; ties retain their average rank.

## `sample_covariance`

```python
sample_covariance(samples: 'ArrayLike', *, ddof: 'int' = 1, assume_centered: 'bool' = False) -> 'CovarianceEstimate'
```

Estimate covariance from samples (observation,coordinate).

ddof selects n-ddof normalization; assume_centered=True uses zero rather than
an estimated mean. Return an owned CovarianceEstimate.

## `ContinuousMarginal`

Scalar marginal law providing CDF, quantile and log density for copula composition.

### `ContinuousMarginal.cdf`

```python
cdf(self, values: 'ArrayLike', /) -> 'ArrayLike'
```

### `ContinuousMarginal.log_density`

```python
log_density(self, values: 'ArrayLike', /) -> 'ArrayLike'
```

### `ContinuousMarginal.quantile`

```python
quantile(self, probability: 'ArrayLike', /) -> 'ArrayLike'
```

## `ConvergenceError`

```python
ConvergenceError(result: 'CovarianceEstimate | StudentTResult | TailFit | MatrixRepair') -> 'None'
```

Raised when a checked numerical method does not converge; result retains the attempted solution and diagnostics.

## `Copula`

```python
Copula(adapter: 'CopulaAdapter') -> None
```

Checked dependence law on unit-interval coordinates with explicitly advertised capabilities.

### `Copula.cdf`

```python
cdf(self, uniforms: 'ArrayLike', *, rng: 'np.random.Generator | None' = None)
```

Elliptical adapters require rng for randomized SciPy integration.

### `Copula.density`

```python
density(self, uniforms) -> 'FloatArray'
```

### `Copula.log_density`

```python
log_density(self, uniforms: 'ArrayLike')
```

### `Copula.sample`

```python
sample(self, size: 'SampleSize' = (), *, rng: 'np.random.Generator')
```

### `Copula.tail_dependence`

```python
tail_dependence(self, i: 'int' = 0, j: 'int' = 1)
```

Pairwise (lower, upper) limiting conditional exceedance probabilities.

## `CopulaAdapter`

Dependence adapter declaring dimension and supported operations on unit-interval coordinates.

### `CopulaAdapter.evaluate`

```python
evaluate(self, operation: 'str', *args: 'object', **kwargs: 'object') -> 'ArrayLike'
```

## `CopulaDistribution`

```python
CopulaDistribution(copula: 'Copula', marginals: 'Sequence[ContinuousMarginal]') -> 'None'
```

Sklar composition. Continuous marginals supply cdf, quantile, log_density.

### `CopulaDistribution.cdf`

```python
cdf(self, values: 'ArrayLike', *, rng: 'np.random.Generator | None' = None)
```

### `CopulaDistribution.density`

```python
density(self, values: 'ArrayLike') -> 'FloatArray'
```

### `CopulaDistribution.log_density`

```python
log_density(self, values: 'ArrayLike') -> 'FloatArray'
```

### `CopulaDistribution.sample`

```python
sample(self, size: 'SampleSize' = (), *, rng: 'np.random.Generator') -> 'FloatArray'
```

## `CopulaFit`

```python
CopulaFit(copula: 'Copula', nobs: 'int', log_likelihood: 'float', repair_distance: 'float' = 0.0, diagnostics: 'Diagnostics' = Diagnostics(backend='user', converged=True, message='', iterations=None)) -> None
```

Fitted copula and fit diagnostics; marginal distributions are not inferred by this result.

## `CovarianceEstimate`

```python
CovarianceEstimate(location: 'ArrayLike', covariance: 'ArrayLike', *, nobs: 'int | None' = None, effective_nobs: 'float | None' = None, shrinkage: 'float | None' = None, diagnostics: 'Diagnostics | None' = None) -> 'None'
```

Location and covariance in feature order, with observation counts and estimation diagnostics.

### `CovarianceEstimate.mahalanobis`

```python
mahalanobis(self, values: 'ArrayLike') -> 'FloatArray'
```

Squared distances; singular covariance has no implicit pseudoinverse.

### `CovarianceEstimate.variance`

```python
variance(self, weights: 'ArrayLike') -> 'FloatArray'
```

Variance of linear combinations, weights shaped (...,d).

## `CovarianceEstimator`

Fit location/covariance from observation rows, returning a checked CovarianceEstimate.

### `CovarianceEstimator.fit`

```python
fit(self, samples: 'ArrayLike') -> "'CovarianceEstimate'"
```

## `Diagnostics`

```python
Diagnostics(backend: 'str' = 'user', converged: 'bool' = True, message: 'str' = '', iterations: 'int | None' = None) -> None
```

Backend termination diagnostics. Inspect converged and message before using a fitted approximation; iterations may be unavailable.

## `FactorCovariance`

```python
FactorCovariance(loadings: 'ArrayLike', factor_covariance: 'ArrayLike', residual_covariance: 'ArrayLike', *, location: 'ArrayLike | None' = None, nobs: 'int | None' = None, diagnostics: 'Diagnostics | None' = None) -> 'None'
```

Cov(X) = B Cov(F) B.T + Cov(epsilon), with uncorrelated factors/noise.

### `FactorCovariance.mahalanobis`

```python
mahalanobis(self, values: 'ArrayLike') -> 'FloatArray'
```

Squared distances; singular covariance has no implicit pseudoinverse.

### `FactorCovariance.variance`

```python
variance(self, weights: 'ArrayLike') -> 'FloatArray'
```

Variance of linear combinations, weights shaped (...,d).

## `MatrixRepair`

```python
MatrixRepair(original: 'ArrayLike', matrix: 'ArrayLike', *, diagnostics: 'Diagnostics | None' = None) -> 'None'
```

Repaired matrix and convergence diagnostics; repair changes the supplied matrix and must be inspected.

## `PeaksOverThresholdResult`

```python
PeaksOverThresholdResult(threshold: 'float', tail: 'TailFit', nobs: 'int') -> None
```

Threshold, fitted excess law and sample count needed to interpret tail exceedance probabilities.

### `PeaksOverThresholdResult.quantile`

```python
quantile(self, probability: 'ArrayLike') -> 'FloatArray'
```

### `PeaksOverThresholdResult.return_level`

```python
return_level(self, period: 'ArrayLike') -> 'FloatArray'
```

Expected exceedance interval in original observation units.

### `PeaksOverThresholdResult.survival`

```python
survival(self, values: 'ArrayLike') -> 'FloatArray'
```

## `StudentTResult`

```python
StudentTResult(location: 'ArrayLike', shape: 'ArrayLike', df: 'float', *, nobs: 'int', log_likelihood: 'float', diagnostics: 'Diagnostics | None' = None) -> 'None'
```

Fitted multivariate Student-t parameters and diagnostics; scale and covariance are distinct quantities.

## `TailFit`

```python
TailFit(family: 'str', xi: 'float', location: 'float', scale: 'float', nobs: 'int', log_likelihood: 'float', diagnostics: 'Diagnostics' = Diagnostics(backend='user', converged=True, message='', iterations=None)) -> None
```

xi uses the extreme-value convention (SciPy genextreme shape is -xi).

### `TailFit.return_level`

```python
return_level(self, period: 'ArrayLike') -> 'FloatArray'
```

GEV return period measured in supplied blocks; GPD in excess events.

## `fit_covariance`

```python
fit_covariance(samples: 'ArrayLike', estimator: 'CovarianceEstimator', *, check_convergence: 'bool' = True) -> 'CovarianceEstimate'
```

Fit any compatible tool and check the returned mathematical state.
