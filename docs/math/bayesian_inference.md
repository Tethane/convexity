# Bayesian inference and parameter uncertainty

Family 11 is implemented in `convexity.math.bayesian`. Priors and analytic
posteriors use the existing `distributions.Distribution` contract. Numerical
likelihoods compose with `estimation.Likelihood`; sequential latent inference
composes with family 10's state-space models. PyMC performs MCMC and variational
inference. No backend is imported merely by importing this family.

## Completed catalog

| Capability | Public entry points | Scope |
|---|---|---|
| Prior laws | `beta`, `gamma`, `inverse_gamma`, existing distributions | Proper scalar/vector laws; independent and dependent prior composition |
| Likelihood/posterior evaluation | `BayesianProblem` | Vector parameter prior plus complete scalar log likelihood, or existing `estimation.Likelihood`; unnormalized log posterior |
| Symbolic Bayesian models | `PyMCModel` | User factory returns a fresh native PyMC model; arbitrary named priors, observations, deterministic variables and latent states |
| Exact conjugate updates | `BetaBinomial`, `GammaPoisson`, `NormalMean`, `NormalInverseGamma`, `GaussianLinearRegression` | Immutable batch or sequential updates; full parameter uncertainty |
| Posterior predictions | Conjugate `.predictive`, `posterior_predictive`, `PosteriorSamples.posterior_predictive` | Integrate joint parameter draws and conditional observation noise; native PyMC out-of-sample models |
| MCMC | `MCMC` | PyMC automatic NUTS/compound steps, explicit NUTS, Metropolis; numerical callbacks use continuous random-walk Metropolis |
| Variational inference | `VariationalInference` | PyMC mean-field ADVI and full-rank ADVI for differentiable continuous models |
| Joint state/parameter inference | `PyMCModel`, `StateParameterModel`, `JointParticleFilter` | Batch joint latent-state/parameter MCMC; sequential bootstrap SMC with static parameters and conditional state priors |
| Joint posterior laws and diagnostics | `PosteriorSamples`, `InferenceDiagnostics`, `JointFilterResult` | Read-only samples, equal-tail credible intervals, empirical joint laws; rank R-hat, bulk/tail ESS, divergences, VI loss; sequential weights/ESS/evidence |
| User inference algorithms | `InferenceMethod`, `infer` | Structural `.fit(model, rng=...)` protocol, returning checked `PosteriorSamples` |

Backend references: [PyMC sampling](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample.html),
[ADVI](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.fit.html),
[posterior prediction](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample_posterior_predictive.html),
and [particles](https://particles-sequential-monte-carlo-in-python.readthedocs.io/en/latest/).

## Installation and examples

Exact conjugate inference needs only the standard project dependencies. Install
optional numerical inference backends with:

```bash
python -m pip install -e '.[dev,bayesian]'
python docs/math/examples/bayesian.py
python docs/math/examples/bayesian_pymc.py
python docs/math/examples/bayesian_sequential.py
python -m pytest -q tests/math/bayesian
```

The extra uses PyMC 5, ArviZ diagnostics, and the same upstream `particles`
revision as [family 10](state_space.md#installation-and-execution). Backend
installation may download packages; inference and all generated-data examples
run locally. The [worked notebook](notebooks/bayesian.ipynb) illustrates
posterior contraction, predictive uncertainty and correlated regression predictions.

## Exact updates and conventions

Conjugate objects describe both priors and posteriors. `prior.update(data)` returns
a new object of the same class; `.posterior` returns its normalized parameter law.
`.predictive(...)` describes future observations after integrating parameter
uncertainty. Missing values are rejected by conjugate updates; omit absent rows
explicitly. Scalar data are supplied as a one-element vector.

```python
import numpy as np
from convexity.math import bayesian as b

posterior = b.BetaBinomial(2, 2).update([7, 6, 8], trials=10)
print(posterior.posterior.quantile([.025, .975]))
print(posterior.predictive(trials=10).mean())

normal = b.NormalInverseGamma(mean=0, precision=1, shape=2, scale=1)
normal = normal.update([1., 2., 1.5, .8])
print(normal.posterior.sample(5, rng=np.random.default_rng(1)))  # (mean, variance)
print(normal.predictive().quantile([.025, .975]))
```

- `BetaBinomial(alpha,beta)` uses independent binomial count rows; scalar trial
  counts broadcast. The predictive law is beta-binomial, including Bernoulli for
  one future trial. A future count of zero trials is deterministic zero.
- `GammaPoisson(shape,rate)` uses Poisson counts with positive known exposures:
  `count_i | rate_parameter ~ Poisson(exposure_i * rate_parameter)`. Gamma uses
  **rate**, not scale. Its predictive law is negative binomial, including parameter
  uncertainty, for one specified future exposure.
- `NormalMean(mean,variance,observation_variance)` uses known Gaussian observation
  variance. Posterior variance is `1/(1/prior_variance+n/observation_variance)`.
  Predictive variance adds observation variance.
- `NormalInverseGamma(mean,precision,shape,scale)` means
  `variance ~ InverseGamma(shape,scale)` and
  `mean_parameter | variance ~ Normal(mean, variance/precision)`.
  The joint parameter order is `(mean,variance)`; a mean marginal and variance
  marginal are also available. The mean marginal and predictive observation law
  are Student-t with `2*shape` degrees of freedom. Mean and variance parameters
  are dependent even when their covariance is zero. Undefined/infinite moments
  retain the distribution family's NaN/Inf conventions.
- `GaussianLinearRegression(mean,covariance,observation_variance)` uses a Gaussian
  coefficient prior and known independent noise variance. The design is `(n,p)`;
  responses are `(n,)`. Insert an intercept column explicitly. Predictive
  covariance is `X_new @ covariance @ X_new.T + observation_variance*I`.
  `include_noise=False` returns the latent regression mean law, which may be
  singular for repeated rows. Prediction shares coefficient uncertainty across
  query rows; it is a joint law, not independent pointwise intervals.

Empty observation vectors leave scalar conjugate priors unchanged. Fractional
counts, invalid exposures, nonpositive variances, nonfinite data, and invalid
shapes fail at the public boundary. Arrays are copied and made read-only.

## MCMC, VI, and custom likelihoods

A native model factory builds a fresh model per inference call:

```python
import pymc as pm

def build():
    with pm.Model() as model:
        mean = pm.Normal('mean', 0, 2)
        scale = pm.HalfNormal('scale', 1)
        pm.Normal('observed', mean, scale, observed=[.8, 1.4, 1.1, .9])
    return model

model = b.PyMCModel(build)
posterior = b.infer(model, b.MCMC(draws=1000, tune=1000, chains=4),
                    rng=np.random.default_rng(2))
print(posterior.mean('mean'), posterior.interval('mean'))
print(posterior.diagnostics.rhat, posterior.diagnostics.ess_bulk)
approximate = b.VariationalInference(method='fullrank_advi').fit(
    model, rng=np.random.default_rng(3))
replicated = posterior.posterior_predictive(rng=np.random.default_rng(4))
```

`samples[name]` is `(chain,draw,*event_shape)` and excludes warmup. A scalar
parameter has `(chain,draw)` shape, a coefficient vector adds its coefficient
axis, and latent trajectories retain their time and state axes. VI outputs use
one leading chain axis solely for shape consistency. All named variables share
the same leading axes; flattening them together preserves posterior dependence.
`.distribution(names=[...])` returns a vector empirical law in the requested
variable order, flattening each event in C order. It does not estimate a
continuous density. `.interval(name,level=.95)` is an equal-tail marginal credible
interval with shape `(2,*event_shape)`, not a simultaneous band or frequentist
confidence interval.

MCMC reports ArviZ rank-normalized R-hat and bulk/tail ESS when enough draws are
available. R-hat is absent for one chain. Undefined diagnostics (e.g. constant
variables) remain NaN. Divergences are reported when the step supplies them;
Metropolis has no divergence statistic. Sampling completion does not assert
convergence. VI reports its negative ELBO loss history and does not report MCMC
convergence statistics or claim that its covariance is exact. ADVI rejects
latent discrete variables. Automatic PyMC MCMC supports discrete/continuous
compound steps; explicit NUTS requires a differentiable model.

For an existing numerical likelihood, supply a proper vector prior and a finite
initial point. Scalar parameters retain their one-coordinate vector shape:

```python
from convexity.math import distributions as dist, estimation as est
likelihood = est.Likelihood([.8, 1.4, 1.1],
    lambda theta, y: dist.normal(theta[0], 1).log_density(y))
problem = b.BayesianProblem(dist.independent([dist.normal(0, 2)]),
                            likelihood, names=('mean',), initial=[0.])
posterior = b.MCMC(method='metropolis').fit(problem, rng=np.random.default_rng(5))
```

`BayesianProblem.log_posterior` is unnormalized: log prior plus complete log
likelihood. It is not model evidence or a normalized posterior density. Callbacks
must return a scalar (or use the existing per-observation `Likelihood` contract).
Support violations may return `-inf`; NaN and `+inf` fail explicitly. Outside
prior support, the likelihood is not evaluated. Priors apply in the declared
parameter coordinates; if the caller changes coordinates, the appropriate
Jacobian belongs in the model. Numerical callbacks have no symbolic gradient,
so they use PyMC Metropolis with continuous priors, not NUTS or ADVI. Use native
PyMC models for discrete parameters and symbolic transforms.

All stochastic public calls require a `numpy.random.Generator`. The adapters
consume seeds from it; repeated calls with fresh equal seeds are reproducible
within the same backend environment. `cores=1` is the default for predictable
local execution. Heavy optional dependencies are imported only when called.

## Posterior prediction

The standalone `posterior_predictive(posterior,conditional,draws=...,rng=...)`
accepts a parameter `Distribution` or `PosteriorSamples`. For the latter,
`conditional` receives a name-to-event mapping from a single joint draw. It must
return a scalar/vector observation `Distribution`. Each generated observation
uses both a parameter draw and a draw from that conditional law. The output is
an empirical predictive law; no Gaussian shape is silently fitted.

Native `.posterior_predictive()` returns forward draws for observed variables.
For new inputs or a new observation shape, supply a fresh `PyMCModel` through
`model=...` and explicit output `names`. Fitted parameter names and shapes must
match the training model. Requesting a fitted free parameter as a prediction is
rejected because PyMC would resample it; access its posterior samples directly.
Numerical likelihood callbacks alone do not define forward observations: use
the standalone conditional-law function for those models.

## Joint state and parameter inference through time

For batch inference, include latent states and unknown parameters in the same
native PyMC model (e.g. a Gaussian random walk plus unknown drift/noise). Joint
samples then integrate parameter uncertainty into state posteriors. Tests compare
a coupled latent-state/offset model with dense joint-Gaussian conditioning.

`StateParameterModel(prior,model_factory,reference,parameter_names=...)` defines
sequential inference. The vector parameter prior can be discrete or continuous.
The factory receives a parameter vector and returns an existing family 10
`StateSpaceModel`. The reference vector checks dimensions and time conventions;
it is never used as a plug-in estimate. Each parameter draw samples from its own
conditional initial state law. Initial states live at observation row zero,
without an extra initial transition, following family 10's timing contract.

`JointParticleFilter.filter` runs the established `particles` bootstrap backend
on augmented `(parameters,states)` particles. Static parameters remain constant
along each propagated trajectory. Resampling selects complete paired particles,
so state/parameter dependence is preserved. Conditional models must share state
and observation dimensions, state names, and time mode/units. Missing observation
coordinates, fully missing rows, and elapsed timestamps follow family 10's
contracts. A fully missing row contributes zero likelihood.

`JointFilterResult.joint_posterior(t)`, `.parameter_posterior(t)`, and
`.state_posterior(t)` expose causal weighted empirical laws. `.filtering` retains
observations, weights, per-time ESS, resampling flags, and log normalizing-constant
estimates. The latter are Monte Carlo log evidence estimates, not unbiased log
evidence. `.smooth(...,paths=...)` traces ancestral joint paths and exposes
smoothed state marginals. One static parameter is retained along each path.

The algorithm does **not** jitter static parameters or invent parameter dynamics.
This finite bootstrap catalog can lose parameter diversity and ancestral path
diversity over long series. ESS describes current weights; it does not count
unique surviving parameters. Inspect both when using long histories. SMC²,
particle MCMC rejuvenation, backward density smoothing for static parameters,
and streaming checkpoint/resume are outside this catalog. The adapter processes
an entire supplied series sequentially; appending data requires a new run. The
underlying particles adapter isolates and restores its legacy global NumPy RNG
as documented in [family 10](state_space.md).

## Acceptance checks

Tests verify exact conjugate formulas and agreement between batch and split
updates; Student-t/negative-binomial predictive moments; joint NIG sampling;
dense Gaussian regression conditioning and shared predictive dependence; actual
PyMC NUTS, Metropolis and both ADVI modes against analytic posteriors; discrete
sampling; posterior predictive shapes and variance; joint latent-state MCMC;
sequential joint particle marginals/evidence against finite-state enumeration;
missing rows, irregular times, static path ancestry, ownership, shape validation,
user inference methods, lazy imports, seeds and global-RNG restoration.
