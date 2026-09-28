# Covariance, dependence and tails

Family 5 lives in `convexity.math.dependence`. It accepts supplied arrays and
mathematical specifications. NumPy, SciPy, scikit-learn and statsmodels perform
the numerical work. No market data, calendars, forecasts or trading objects are
needed. Existing family-2 distributions compose directly with this family.

## Finite acceptance catalog

| Mathematical task | Public objects and implementation |
|---|---|
| Sample covariance | `SampleCovariance`, `sample_covariance`; scikit-learn empirical covariance, explicit `ddof` and centering |
| Exponential weighting | `EWMACovariance`, `ewma_covariance`; normalized finite-history geometric weights, weighted location, bias correction and effective sample size |
| Shrinkage | `ShrunkCovariance`, `LedoitWolf`, `OAS`; scikit-learn estimators, with fitted shrinkage exposed |
| Factor covariance | `FactorCovariance`, `fit_factor_covariance`; supplied loadings/covariances or SciPy least-squares exposures to observed factors |
| Correlation | `correlation`, `covariance_to_correlation`, `correlation_to_covariance`; Pearson, Spearman and Kendall tau-b |
| PSD constraints | `PositiveSemidefinite`, `MatrixRepair`, `nearest_correlation`; SciPy spectral projection and statsmodels fixed-diagonal correlation projection |
| Multivariate heavy tails | `multivariate_student_t`; reuse family-2 laws; `StudentT`, `fit_student_t`, `StudentTResult` fit location, positive-definite shape and optionally degrees of freedom with SciPy |
| Copula laws | `IndependenceCopula`, `GaussianCopula`, `StudentTCopula`, bivariate `ClaytonCopula` and `GumbelCopula`; statsmodels densities/sampling and SciPy elliptical CDFs |
| Copula estimation/composition | `pseudo_observations`, `KendallCopula`, `fit_copula`, `CopulaFit`, `CopulaDistribution`; rank estimation and continuous marginal composition |
| Extreme-value laws | `generalized_pareto`, `generalized_extreme_value`; SciPy laws wrapped in family 2's public `Distribution` interface |
| Extreme-value fitting | `GeneralizedPareto`, `GeneralizedExtremeValue`, `fit_gpd`, `fit_gev`, `TailFit`; SciPy likelihood fits with optimizer diagnostics |
| Upper-tail modeling | `block_maxima`, `peaks_over_threshold`, `PeaksOverThresholdResult`; explicit block alignment, exceedance rates and return-period units |
| Extensibility | `CovarianceEstimator`, `fit_covariance`, `CopulaAdapter`, `Copula`, `ContinuousMarginal`; structural protocols without registration or required subclassing |

This catalog covers static dependence, elliptical Student-t heavy tails and
univariate stationary extreme-value models. Dynamic conditional covariance,
nonlinear shrinkage, sparse precision estimation, vine copulas, automatic
threshold/block selection, declustering, nonstationary extremes and fit
uncertainty are outside it. Rank copula fitting does not estimate Student-t df;
the multivariate Student-t likelihood fitter can estimate it.

Run the [problem-book script](../examples/dependence.py) or the
[worked notebook](../notebooks/dependence.ipynb). Both contain generated
inputs and numerical assertions; the notebook also visualizes the models.

## Layers, shapes and ownership

`interfaces/` owns mathematical state, validation boundaries and extension
contracts. `_backends/` contains library adapters. `builtins/` provides immutable
specifications and convenience functions. All public names are exported by the
family. Importing the family loads only NumPy; numerical backends and family-2
distributions load on use. No new dependency is needed.

| Quantity | Shape and convention |
|---|---|
| Covariance or copula training samples | `(n,d)`, rows are observations, columns retain caller order |
| Univariate covariance/Student-t training | `(n,1)` |
| Location / covariance | `(d,)` / `(d,d)` |
| Factor loadings / factor observations | `(d,k)` / `(n,k)` |
| Factor covariance / residual covariance | `(k,k)` / `(d,d)`; a `(d,)` residual input means diagonal variances |
| Law evaluation / copula evaluation | `(...,d)` returns the leading shape `(...)` |
| Sampling | integer or tuple `size` gives `size + event_shape`; `size=()` gives one event |
| Extreme-value data | `(n,)`; each row is an excess or a block maximum |

Data must be finite real numeric values. Empty training inputs, NaNs, infinities,
complex numbers, booleans and numeric strings are rejected. There is no pairwise
deletion, imputation, calendar inference, scaling or annualization. Fits use only
the rows supplied by the caller. Covariance units are products of input units;
correlations and copula coordinates are dimensionless.

Results copy arrays to read-only float64 storage. Outputs are read-only too.
Specifications produce fresh fitted state and retain no mutable library
estimator. Built-in laws use the supplied `numpy.random.Generator`; random
sampling never takes an implicit seed. Zero-sized sample batches are supported.

`Diagnostics` records backend, convergence, message and available iteration
count. Iterative fits and projections raise `ConvergenceError` on failure and
retain `.result`; `check_convergence=False` returns failed state for inspection.
Malformed data or an invalid final law raises a validation/library error instead.
Successful local optimization does not establish a global optimum, a unique
model or statistical coverage. No fitted parameter covariance is fabricated.

## Covariance estimates and shrinkage

```python
import numpy as np
from convexity.math import dependence as dep

x = np.random.default_rng(42).normal(size=(200, 4))
empirical = dep.sample_covariance(x, ddof=1)
weighted = dep.ewma_covariance(x, decay=.97, ddof=1)
shrunk = dep.LedoitWolf().fit(x)
variance = shrunk.variance(np.ones(4) / 4)
```

`SampleCovariance` estimates the column mean and divides centered cross-products
by `n-ddof`, with `ddof` either 0 or 1. At least two observations are required.
`assume_centered=True` fixes location to zero and computes second moments, with
the same explicit divisor. The caller owns the zero-mean assumption.

EWMA takes oldest-to-newest rows. It uses normalized weights
`w[i] = decay**(n-1-i) / sum(decay**arange(n))`, with `0 < decay <= 1`.
The covariance is `sum(w[i] * outer(x[i]-mu, x[i]-mu)) / (1-ddof*sum(w**2))`.
The mean is weighted unless fixed to zero. `effective_nobs = 1/sum(w**2)`;
the effective count measures weight concentration, not serial independence.
`ddof=1` is the weighted mean-estimation correction for independent observations.
`ddof=0` is the default. `decay=1` recovers equal weighting. A degenerate effective
count rejects bias correction. This is a finite-history weighted estimate, not
an initialized streaming volatility forecast.

All three shrinkage fits use empirical `ddof=0` moments and the spherical target
`trace(S)/d * I`. Their result is `(1-alpha)*S + alpha*trace(S)/d*I`.
`ShrunkCovariance(alpha)` supplies alpha; `LedoitWolf` and `OAS` estimate it using
their library methods. The fitted `.shrinkage` is exposed. No standardization is
implicit; changing coordinate units changes the target. Shrinkage does not
guarantee positive definiteness for zero total variance or zero fitted alpha.

`CovarianceEstimate` exposes `location`, `covariance`, `dimension`, `nobs`,
optional `effective_nobs` and `shrinkage`, plus `diagnostics`. Singular PSD
covariances are valid. `.precision` requires numerical positive definiteness;
there is no silent pseudoinverse. `.mahalanobis(x)` returns squared distances.
`.variance(weights)` evaluates linear-combination variances over any leading
batch axes. `.rank`, `.standard_deviation` and `.correlation` are inspectable.

## Factor covariance and correlations

`FactorCovariance(B, F, R)` represents `B @ F @ B.T + R`, assuming uncorrelated
factors and residuals. Both covariance inputs may be PSD, including zero noise.
Loadings need not be orthogonal. Optional `location` defaults to zero.
`common_covariance` exposes the first term. Independent rotations of factor
coordinates, with the matching transformed `F`, leave observation covariance
unchanged.

Family-4 models compose without refitting: pass a `GaussianFactorModel` or
`FactorAnalysisResult` object's `.loadings`, identity factor covariance,
`.noise_variance` and `.location` into this constructor.

`fit_factor_covariance(x, factors)` fits centered OLS exposures and uses the same
`n-ddof` divisor for factor and residual moments. Factors must have full centered
column rank and leave residual degrees of freedom. The result's location is the
observation mean; no factor forecasting/intercept model is implied.
`residual='diagonal'` discards cross-residual covariance as an explicit modeling
assumption. `residual='full'` reproduces sample covariance by OLS orthogonality.
These residual moments are covariance decomposition terms, not unbiased
regression-noise estimates with an `n-k-1` divisor.

`correlation(x, method='pearson'|'spearman'|'kendall')` uses all rows jointly.
Spearman uses average ranks; Kendall uses tau-b, including tie adjustment.
These statistics differ from a latent copula correlation parameter.
Zero-variance columns have undefined correlations and are rejected.
`correlation_to_covariance` takes standard deviations, not variances; zero
standard deviations are permitted, though reverse conversion is then undefined.

## Positive-semidefinite constraints

`PositiveSemidefinite(min_eigenvalue=0, tolerance=1e-10).contains(A)` checks a
symmetric matrix with a scale-relative tolerance. Invalid shapes/asymmetry raise
errors. `.project(A)` uses SciPy eigenvectors to clip eigenvalues to the supplied
nonnegative floor. It is the Frobenius projection for a symmetric input; its
diagonal can change. The floor has the same units as the matrix.

`nearest_correlation(A, min_eigenvalue=1e-10)` requires a unit-diagonal symmetric
input, which may be indefinite or have entries outside `[-1,1]`. It uses
statsmodels' alternating projection and retains the diagonal. It returns an
approximate constrained projection, not a claim of exact arithmetic.
The library can exhaust its budget at an active constraint. The adapter then
compares the final two library iterates at relative Frobenius tolerance `1e-10`
and checks the eigenvalue floor to absolute tolerance `1e-10`. Otherwise it
reports nonconvergence. `maxiter` defaults to 1000.

Both operations return `MatrixRepair` with `.matrix`, Frobenius `.distance`,
minimum eigenvalues before/after and diagnostics. A failed repair may still be
indefinite. Constructors never silently repair an indefinite covariance.
PSD validation permits scale-relative roundoff of `1e-10` and rejects negative
diagonal variances; a tolerance does not express an economic risk constraint.

## Multivariate Student-t

```python
law = dep.multivariate_student_t(6, [0., 0.], [[1., .5], [.5, 1.]])
draws = law.sample(2000, rng=np.random.default_rng(7))
fitted = dep.StudentT(df=5., estimate_df=True).fit(draws)
conditional = fitted.distribution.condition([0], [1.])
```

The law is the completed family-2 Student-t, including density, sampling,
moments, marginals and conditioning. Shape is **not covariance**:
`Cov(X) = shape * df/(df-2)` when `df>2`. `StudentTResult.covariance` raises if
there is no finite covariance. A diagonal shape does not make t coordinates
independent: they share the radial scale.

`StudentT` fits location and a full positive-definite shape using SciPy
L-BFGS-B on a Cholesky parameterization. Diagonal Cholesky entries use log
coordinates. It requires `n>d` and full centered column rank. With
`estimate_df=False`, supplied df stays fixed; with `True`, it is optimized in
log coordinates within `df_bounds=(.5,200.)`. Boundary estimates are reported in
diagnostics. Default tolerance is `1e-10`, maximum iterations 1000. The result
retains location, shape, df, sample count and summed log likelihood. Ill-scaled
data or nearly singular fits may require explicit rescaling by the caller.

## Copulas and continuous marginal composition

Gaussian and Student-t copulas accept a positive-definite `(d,d)` correlation
matrix with `d>=2`, checked against SciPy's numerical rank convention at
construction. Student-t additionally takes finite positive df. Clayton
requires `theta>0`; Gumbel requires `theta>=1`, with 1 handled as independence.
Both Archimedean adapters are bivariate and support positive dependence only.
`IndependenceCopula(d)` works in any dimension `d>=2`.

Laws expose `log_density`, `density`, `cdf`, `sample`, `capabilities` and pairwise
`tail_dependence(i=0,j=1)`, returning `(lower,upper)`. Densities require interior
uniform coordinates `(0,1)`. CDFs accept the closed cube `[0,1]`; boundary faces
reduce to marginal laws. Gaussian/t CDFs require an explicit generator because
SciPy integrates them numerically. They are reproducible numerical estimates,
not exact orthant probabilities. Student-t CDFs use SciPy directly where the
statsmodels method is unavailable.

Gaussian and independence copulas have zero asymptotic pair tail dependence.
The Student-t coefficient is
`2 * t_cdf(-sqrt((df+1)*(1-rho)/(1+rho)), df+1)` in both tails. This adapter uses
the formula through SciPy, including compatibility with older statsmodels
versions. Clayton has lower coefficient `2**(-1/theta)` and upper zero; Gumbel
has lower zero and upper `2-2**(1/theta)`.

`pseudo_observations(x)` computes average column ranks divided by `n+1`.
`KendallCopula(family='gaussian').fit(x)` and `fit_copula` delegate pairwise
tau inversion to statsmodels. For `family='student_t'`, df must be supplied;
Kendall tau cannot identify it. Clayton/Gumbel reject fits outside their supported
parameter domains. Constant columns are rejected. Ties retain average ranks;
this is not a discrete-data copula likelihood or a tie-corrected inferential fit.
The reported summed `log_likelihood` is evaluated on pseudo-observations.

Pairwise latent correlations can be indefinite or singular. They are rejected
by default; `repair_correlation=True` explicitly projects with eigenvalue floor
`1e-8` and records the Frobenius `repair_distance`. The repaired law need not
reproduce every fitted pairwise rank correlation exactly.

```python
from convexity.math import distributions as dist

joint = dep.CopulaDistribution(
    dep.StudentTCopula([[1., .6], [.6, 1.]], df=5),
    [dist.normal(), dist.student_t(7)])
scenarios = joint.sample(1000, rng=np.random.default_rng(8))
log_density = joint.log_density(scenarios[:10])
```

`CopulaDistribution` implements `C(F_1(x_1),...,F_d(x_d))`, inverse-marginal
sampling, and `log c(F(x)) + sum(log f_j(x_j))`. Its marginals may be family-2
laws or user objects with scalar `event_shape=()`, `measure='continuous'`, and
shape-preserving `cdf`, `quantile`, `log_density`. Discrete marginals are rejected.
Joint covariance, quantiles and conditioning are not claimed generically.
Outside marginal support the density is zero. A numerically saturated marginal
CDF at a positive-density point raises an interior-probability error rather
than substituting an arbitrary clipped copula density. Samples at floating-point
probability endpoints can likewise expose a marginal's infinite quantile.

## Extreme values, thresholds and return periods

Both law constructors return a family-2 `Distribution`. They expose log density,
CDF, stable survival probability, quantiles, sampling, support and moments.
They use the conventional extreme-value index `xi`: positive is heavy-tailed,
zero is exponential/Gumbel, negative gives a finite upper endpoint.
SciPy's GEV shape parameter is `c=-xi`; SciPy's GPD shape is `c=xi`.

`generalized_pareto(xi, scale=1, threshold=0)` is a shifted excess law.
`fit_gpd(excesses)` / `GeneralizedPareto().fit(excesses)` fixes location to zero
and requires at least three nonconstant, nonnegative observations.
`generalized_extreme_value(xi, loc=0, scale=1)` is a law for block maxima.
`fit_gev(maxima)` / `GeneralizedExtremeValue().fit(maxima)` requires at least
four nonconstant maxima. SciPy's fits use a monitored Nelder-Mead optimizer,
default maximum 3000 iterations and parameter/objective tolerance `1e-8`.
The fitted support and every observation's log density are checked.
The likelihood is unconstrained in xi and local; endpoint pathologies can
prevent a regular or globally bounded MLE, especially with small samples.

`block_maxima(x, block_size)` uses nonoverlapping positional blocks and rejects
an incomplete final block. Opt in to dropping it with `drop_incomplete=True`.
The caller selects meaningful equal-exposure blocks; there is no calendar
inference. `TailFit.return_level(T)` solves survival `1/T`, with `T>1` in blocks
for GEV and excess events for a bare GPD fit.

`peaks_over_threshold(x, threshold)` fits a GPD to **strict** excesses `x>u` and
retains `exceedance_rate = number_of_excesses / nobs`. For `x>=u`, its upper-tail
survival is `exceedance_rate * GPD.sf(x-u)`. It provides quantiles only at
`p>=1-exceedance_rate`; it does not invent a law below the threshold.
Its return period uses original observation units, with `T>=1/exceedance_rate`.
Return levels use inverse survival directly, avoiding loss from `1-1/T` for
large periods. This is an expected exceedance interval; waiting-time and
multi-period event-probability interpretations require additional independence
assumptions. Threshold choice, tail stability and serial clustering remain the
caller's modeling decisions. For lower tails, explicitly transform the sign.

## User extensions and verification

```python
class DiagonalEstimator:
    def fit(self, samples):
        return dep.CovarianceEstimate(
            samples.mean(axis=0), np.diag(samples.var(axis=0, ddof=1)),
            nobs=len(samples))

custom = dep.fit_covariance(x, DiagonalEstimator())
```

An estimator implements `.fit(samples)` and returns `CovarianceEstimate` (or
`FactorCovariance`). `fit_covariance` checks result dimensions, count and
convergence. Custom copulas supply `dimension`, a frozen `capabilities` set,
and `evaluate(operation, *args, **kwargs)` to `Copula`. Operations use validated
read-only points; sampling gets a shape tuple and generator. The wrapper checks
result shapes and probability ranges. User adapters own their state, numerical
accuracy and mathematical claims.

```bash
python -m pytest -q tests/math/dependence
python -m pyright --pythonpath "$(command -v python)"
python examples/dependence.py
```

The suite verifies covariance divisors and weight order, shrinkage formulas,
factor recovery and covariance reconstruction, rank statistics, singular and
indefinite matrices, repair failures, Student-t recovery and missing moments,
copula uniform margins and exact joint-density identities, rank-fit recovery,
CDF boundaries, extreme-value shape signs/support, tail recovery, exceedance
rates, return-period calibration, custom tools, ownership and import isolation.
All 170 family tests pass with the current libraries and the declared minimums:
NumPy 1.26.4, SciPy 1.15.3, scikit-learn 1.6.1 and statsmodels 0.14.4.

Backend references: [scikit-learn covariance](https://scikit-learn.org/stable/modules/covariance.html),
[statsmodels copulas](https://www.statsmodels.org/stable/distributions.html#copulas),
[nearest correlation](https://www.statsmodels.org/stable/generated/statsmodels.stats.correlation_tools.corr_nearest.html),
[SciPy multivariate t](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_t.html),
[GPD](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genpareto.html),
[GEV](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genextreme.html).
