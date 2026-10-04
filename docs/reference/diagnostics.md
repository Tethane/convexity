# `convexity.math.diagnostics`

[Reference index](README.md) · [Contract conventions](../public_contracts.md)

Diagnostics mathematics on supplied arrays and models.

## `TestResult`

```python
TestResult(statistic: 'ArrayLike', pvalue: 'ArrayLike', method: 'str', null: 'str', details: 'Mapping[str, object] | None' = None) -> 'None'
```

Statistical test output with statistic, p-value and method information; interpret under its null assumptions.

## `BootstrapResult`

```python
BootstrapResult(estimate: 'ArrayLike', samples: 'ArrayLike', lower: 'ArrayLike', upper: 'ArrayLike', level: 'float', block_length: 'int') -> 'None'
```

Bootstrap statistic replicates and uncertainty summaries under the requested resampling scheme.

## `Score`

Evaluate predictions against observations using the declared statistical scoring convention.

## `log_score`

```python
log_score(log_density: 'ArrayLike', *, weights: 'ArrayLike | None' = None) -> 'float'
```

Return the negative weighted mean of supplied log probabilities/densities.

Smaller is better; values must refer to the same observation coordinates and measure.

## `quantile_score`

```python
quantile_score(actual: 'ArrayLike', prediction: 'ArrayLike', level: 'float', *, weights: 'ArrayLike | None' = None) -> 'float'
```

Return mean pinball loss at a probability level strictly between zero and one.

Prediction and actual must align; the score uses actual-outcome units.

## `crps`

```python
crps(samples: 'ArrayLike | PathScenarios', actual: 'float', *, weights: 'ArrayLike | None' = None) -> 'float'
```

Scalar empirical-law CRPS, O(n log n), weighted atoms allowed.

## `energy_score`

```python
energy_score(samples: 'ArrayLike | PathScenarios', actual: 'ArrayLike', *, weights: 'ArrayLike | None' = None, beta: 'float' = 1.0) -> 'float'
```

Joint-vector/path energy score; flatten event coordinates explicitly.

## `variogram_score`

```python
variogram_score(samples: 'ArrayLike | PathScenarios', actual: 'ArrayLike', *, weights: 'ArrayLike | None' = None, order: 'float' = 0.5, pair_weights: 'ArrayLike | None' = None) -> 'float'
```

Score a joint vector scenario law against one observed vector.

Pair weights and the positive order weight coordinate differences; marginal
forecasts alone cannot determine this dependence-sensitive score.

## `pit`

```python
pit(cdf: 'ArrayLike', *, lower_cdf: 'ArrayLike | None' = None, rng: 'np.random.Generator | None' = None) -> 'FloatArray'
```

Continuous F(y), or randomized F(y-)+U*(F(y)-F(y-)) for atoms.

## `interval_coverage`

```python
interval_coverage(actual: 'ArrayLike', lower: 'ArrayLike', upper: 'ArrayLike', *, weights: 'ArrayLike | None' = None) -> 'float'
```

Return weighted empirical inclusion frequency for aligned closed intervals.

Marginal coverage is not simultaneous coverage and does not define a joint law.

## `residual_test`

```python
residual_test(residuals: 'ArrayLike', *, method: 'str' = 'ljung_box', lags: 'Sequence[int]' = (1, 5, 10)) -> 'TestResult'
```

Test a scalar residual series for the selected dependence/normality diagnostic.

Return statistics and p-values; lags are observation offsets, not elapsed time.

## `stationarity_test`

```python
stationarity_test(series: 'ArrayLike', *, method: 'str' = 'adf', regression: 'str' = 'c', lags: 'int | None' = None) -> 'TestResult'
```

Run ADF or KPSS on a scalar series with explicit trend/lag specification.

ADF tests a unit-root null; KPSS tests a stationarity null. Interpret p-values
against the selected null, not as posterior model probabilities.

## `cointegration_test`

```python
cointegration_test(dependent: 'ArrayLike', regressors: 'ArrayLike', *, trend: 'str' = 'c', lags: 'int | None' = None) -> 'TestResult'
```

Test a scalar dependent series against aligned regressor columns.

Return the Engle-Granger statistic/p-value; the null is no cointegration.

## `hac_mean_covariance`

```python
hac_mean_covariance(observations: 'ArrayLike', *, lags: 'int') -> 'FloatArray'
```

Bartlett HAC covariance of the sample mean; no degrees-of-freedom correction.

## `compare_scores`

```python
compare_scores(first: 'ArrayLike', second: 'ArrayLike', *, lags: 'int') -> 'TestResult'
```

Paired asymptotic normal comparison; negative difference favors first.

## `adjust_pvalues`

```python
adjust_pvalues(pvalues: 'ArrayLike', *, method: 'str' = 'holm') -> 'FloatArray'
```

Adjust a vector of valid p-values for the selected multiple-testing procedure.

Holm controls familywise error; FDR procedures use their stated dependence assumptions.

## `block_bootstrap`

```python
block_bootstrap(statistic: 'Callable[[FloatArray], ArrayLike]', observations: 'ArrayLike', *, block_length: 'int', draws: 'int' = 999, level: 'float' = 0.95, rng: 'np.random.Generator') -> 'BootstrapResult'
```

Circular fixed-length blocks of rows; percentile intervals, paired columns.

Block length 1 gives IID resampling. Choose blocks to cover dependence and
horizon overlap; intervals do not correct bias or nonstationarity.
