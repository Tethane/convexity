# Adversarial QA: all 36 `convexity.math` families

Audit date: 2026-09-30. Scope: the existing bounded catalog in
[direction.md](direction.md), including public interfaces, implementations,
provider adapters, examples, notebooks and cross-family workflows. No additional
model family was added. The working tree already contained extensive unfinished
version-control bookkeeping; this report concerns the tested filesystem, not a
claim that those files have been committed or released.

## Verdict

The reproduced numerical and API defects listed below are corrected and have
regression coverage. The public-contract follow-up supplies authored docstrings,
explicit ordinary entry-point types, structural extension contracts and a
[generated reference](reference/README.md) for all 36 families. The supported
validation scope is the user's **quant environment on Python 3.14**, not a
multi-version compatibility matrix.

The earlier public-contract blocker is closed for the documented, tested public
workflows. This remains a bounded engineering assessment: dtype annotations do
not prove mathematical shapes, units or arbitrary user callback correctness.
Read the [shared conventions](public_contracts.md) and the limitations below
before interpreting this as a claim about unrestricted numerical inputs,
upstream versions or application-specific model validity.

## Public-contract follow-up

- Ordinary exported functions and concrete constructors have explicit parameter
  and return annotations. Public signatures contain no explicit `Any`. Structural
  protocol constructors are excluded: users supply implementing objects.
- Public callable objects have authored docstrings, including records whose old
  dataclass-generated signature had misleadingly counted as documentation.
  The 36 reference pages include method signatures, overloads and expanded
  `TypedDict` convenience options; a regression checks reference freshness.
- `convexity.math.types` provides shared dtype/callback names. Constructors accept
  `ArrayLike` while result fields expose validated numerical arrays. Explicit
  constructors preserve copying, read-only results, dataclass replacement and
  independent default factories.
- Custom pricing models no longer require a QuantLib conversion method. Native
  conversion has a separate protocol. Generic pricing/optimization/inference
  contracts preserve custom input types. Callback parameter names no longer
  prevent owned distributions from satisfying likelihood/copula protocols.
- State-estimator dispatch retains implementation-specific arguments through
  `ParamSpec`, including particle-method RNGs. Fitted learning, covariance,
  transform and GP results have explicit return types. Scalar/per-horizon scores,
  generalized eigenproblems and automatic-differentiation callbacks have distinct
  overloads where the output/callback contract changes.
- Corrected annotations include raw grid arrays, sparse matrices/operators,
  native financial dates, cash-flow inputs, optional result fields and scalar
  versus array payoffs. Forecast moments, joint covariance and marginal intervals
  now state their axes and probability meaning in code documentation.
- Two runtime gaps received regressions: scenario MPC/liquidation reject an
  optimal solver status without primal arrays, and forecast comparison accepts
  a read-only `Mapping`, as its public contract promises.

Final follow-up verification in **Python 3.14.7 / quant**:

- **2,020 tests passed**, no skips or expected failures; 70 recorded
  provider/numerical warnings (271.23 seconds).
- **Pyright: 0 errors, 0 warnings**, 612 files, configured basic mode.
- **47/47 examples and 35/35 notebooks passed**. The real Chronos-2 checkpoint
  fine-tuning path also passed after the final provider-call typing change.
- **46 public-contract regressions passed**, including generated-reference
  freshness. A further focused run passed 230 constructor/financial/forecast/ODE
  checks during finalization.
- **908 export names / 831 SDK-owned callable entries / 36 families**. No owned
  callable lacks authored documentation; ordinary signatures have no missing
  annotations, explicit `Any`, or bare callback/container parameters. The three
  `Literal` aliases lack runtime `__doc__` values and are documented by their
  permitted values in the reference.
- Focused Ruff import checks and `git diff --check` pass.

Full outcomes, dependency versions and source hashes are in the
[machine-readable evidence](sdk_audit_evidence.json).
The prior coverage figures below are historical measurements from the numerical
audit, not a claim that new constructor/docstring lines have identical coverage.

## Original numerical-audit evidence (before contract follow-up)

- Full final suite: **1,974 passed, no skipped or expected-failure cases**, 70
  provider/numerical warnings, 314.98 seconds, Python 3.14.7.
- Coverage of `convexity.math`: **16,408 / 17,638 statements (93.03%)**;
  **3,955 / 5,070 branches (78.01%)**. This leaves 1,230 statements and 1,115
  branch outcomes unexercised. Coverage is from tests, not padded with example runs.
- Full configured Pyright check: **0 errors, 0 warnings**. Focused Ruff checks
  pass for the new audit scripts, regressions and common numerical helpers.
  Existing broad style findings elsewhere were not misreported as mathematics.
- Python 3.12.3 focused public regressions/composition: **182 passed, 2 skipped**;
  the skips require FilterPy, absent in that environment.
- **47/47 examples and 35/35 notebooks passed** on the numerical-audit source tree,
  using the checked-in verification script and cached pretrained checkpoints.
- Wheel build and import from an isolated installation directory passed:
  **36 families, 900 export names**, optional providers blocked, `py.typed`
  present. SciPy and the other declared core dependencies are still required.
  `pip check` reports no broken requirements in the full environment.

Reproduce in an environment with the documented extras:

```bash
python -m pytest tests -q
python -m pyright --pythonpath "$(command -v python)"
python scripts/build_math_reference.py --check
python scripts/audit_math_sdk.py --output /tmp/convexity-api.json
python scripts/verify_math_examples.py --offline --output /tmp/convexity-examples
python scripts/verify_math_examples.py --notebooks --offline --output /tmp/convexity-notebooks
```

The offline commands require the pretrained checkpoints to be cached. Notebook
execution uses a fresh kernel for each file and does not replace saved outputs.

The machine-readable [evidence](sdk_audit_evidence.json) records versions,
family-level coverage and execution outcomes. The reproducible
[inventory command](../../scripts/audit_math_sdk.py) enumerates public exports and
hashes source files. Imports and coverage are measurements of reach, not proofs
that all possible calls are correct. This audit does not claim exhaustive testing
of every combination of arguments, callback or provider version.

## Defects corrected

| ID | Root cause and observable failure | Correction and regression evidence |
| --- | --- | --- |
| QA-01 | Absolute or whole-matrix covariance tolerances accepted invalid small-unit matrices, negative variances, or a small indefinite block next to a large valid variance. | Shared correlation-coordinate PSD validation now checks every active coordinate, symmetry and zero-variance support. It is used by distributions, estimation, dependence, Bayesian models, forecasts, state space, processes/simulation/rough validation, calibration and decision families. `test_adversarial_contracts.py`, `test_covariance_units.py`, `test_mixed_units.py`. |
| QA-02 | Gaussian provider rank heuristics silently discarded a valid low-variance coordinate. For `diag(1e20, 1)`, the joint log density disagreed with the independent product. Raw eigendecomposition also damaged correlated innovations and chance constraints in mixed units. | Standardized Gaussian factorization, whitening and log-volume evaluation preserve positive coordinates. Singular densities retain the Jacobian on the original affine support. Sampling, state kernels, sigma points, HAR paths and Gaussian chance cones use the same factorization. Analytic density/product, covariance and support tests cover these paths. |
| QA-03 | Nonlinear RTS smoothing used a covariance-wide pseudoinverse cutoff. Rescaling a state could stop future observations from updating that state. | Compute the conditional gain in standardized coordinates, including singular support. EKF and UKF filtering/smoothing must commute with diagonal unit changes; they also retain the linear Kalman reduction tests. |
| QA-04 | Student-t elliptical laws inherited covariance-rank problems; subtracting large log-gamma values loses the normal limit. | Standardized Student-t density/sampling/CDF calculations and a stable gamma-ratio normalization. Tests compare an independent SciPy law in well-scaled coordinates, transformation Jacobians, and the Gaussian limit through `df=1e308`. |
| QA-05 | Hermitian checks used an absolute floor, allowing material asymmetry after a unit change. | Normalize the symmetry comparison before calling the eigensolver. Public `numerical.eigh` rejection tests span `1e-20` to `1e20`. |
| QA-06 | Sorted CRPS formulas subtracted large uncentered sums; a common location shift changed the score or made an exact atom inaccurate. | Center samples on the observation before accumulation, in both diagnostics and forecasting. Compare to the pairwise definition through translations of magnitude `1e16`. |
| QA-07 | Forecast results advertised quantiles for scalar mixtures without an inverse-CDF capability; explicit quantile tables could be shadowed by such laws. | Advertise/invoke marginal inversion only when every scalar law supports it, and preserve explicit quantile-table fallback. Sampling/density capabilities also follow a supplied law's actual capabilities. |
| QA-08 | Duplicate forecast probability representations could contradict one another. Default absolute tolerances accepted radically different tiny variances. | Relative variance agreement and joint-versus-marginal moment checks reject demonstrable contradictions. Equality of arbitrary user-defined full laws remains a documented caller precondition, not a claimed theorem. |
| QA-09 | Summing very large ensemble weights overflowed before normalization. | Scale by the maximum before summing; `[1e308,1e308]` agrees with `[1,1]`. |
| QA-10 | Drift-span checks used an absolute tolerance, accepting a nonzero drift change with zero diffusion. | Scale residuals to the drift difference; impossible equivalent-measure changes are rejected even at drift `1e-20`. |
| QA-11 | Continuous-time migration and HJB generator checks used absolute conservation tolerances. | Relative row-conservation checks, nonpositive generator diagonals and explicit absorbing-default rows. Tests change time units by 40 orders of magnitude. |
| QA-12 | HJB uniformization imposed a rate floor of one, causing failure for zero dynamics and tiny discounts. | Use the actual exit rate and handle zero generators explicitly. The analytic reward/discount solution converges immediately. |
| QA-13 | Stable-Baselines training reseeded Python, NumPy and Torch process-global streams. | Restore Python/NumPy state in `finally` and use Torch RNG isolation, including available CUDA generators. Real A2C and PPO training regressions check repeatability and ambient stream preservation. |
| QA-14 | Discount-curve fitting returned a raw SciPy `OptimizeResult` and ignored the reference date. | Return public `CurveFitResult`, validated Python reference date, owned parameters and Convexity calibration diagnostics. Preserve `.x` as a compatibility alias; failed calibration raises. Generated-rate recovery and round-trip serialization tests cover the result. |
| QA-15 | Installed consumers had no PEP 561 marker; a provider scalar conversion also disagreed with its type stub. | Package `py.typed`, normalize the ARCH scalar explicitly, and check the wheel/type checker. This does not turn remaining `Any` annotations into precise contracts. |

New regressions live in [tests/math/audit](../../tests/math/audit). Existing
Monte Carlo covariance tests had arbitrary absolute thresholds smaller than
ordinary sampling variation for some valid matrix roots. Those thresholds now
use the Gaussian/Wishart covariance standard error, with a five-standard-error
bound; deterministic covariance identities remain separately checked. Seeded
sequences can change after the corrected factorization: repeatability is promised
for the same implementation/environment, not bitwise identity across upgrades.

The implementation choice follows the distinction in SciPy's
[Cholesky covariance representation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.Covariance.from_cholesky.html)
between factor-based whitening and covariance rank inference. Student-t reference
checks use the documented [multivariate-t density](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_t.html).

## Family-by-family mathematical evidence

Each row identifies executable mathematical checks, beyond export/import smoke
tests. `tests/math/` contains the suites; `docs/math/examples/` contains the named scripts.
Several financial and decision families deliberately share integration suites.

| Family | Mathematical checks reviewed/executed | Example |
| --- | --- | --- |
| numerical | Analytic derivatives/integrals, matrix equations, sparse/dense agreement, ODE convergence, roots and solver failure | `numerical_mathematics.py` |
| distributions | Scalar identities, moments, CDF/quantile consistency, conditioning, mixtures, tensor gradients, mixed-unit Jacobians | `distributions.py` |
| estimation | Linear closed forms, likelihood recovery, constraints, rank deficiency, intervals and bootstrap | `statistical_estimation.py` |
| transforms | Scaling/inversion, PCA/whitening covariance, lags/differences and coordinate order | `transforms.py` |
| dependence | Covariance repair, copula independence/tails, extreme-value recovery | `dependence.py` |
| learning | Regression/classification generated-data recovery, quantile loss, pipelines and neural gradients | `predictive_learning.py` |
| gaussian_processes | Kernel identities, Gaussian conditioning, interpolation, singular support and unit-rescaled densities | `gaussian_processes.py` |
| forecasting | AR/VAR identities, generated-data recovery, causal covariates, joint-vs-marginal capabilities, real pretrained execution | `forecasting.py`, `forecasting_advanced.py`, `forecasting_pretrained.py` |
| volatility | ARCH/GARCH recursion, HAR lognormal moments and cross-horizon covariance, variance units | `volatility.py`, `volatility_har.py` |
| state_space | Dense Gaussian conditioning, linear reduction of EKF/UKF, smoothing, missing observations, elapsed time, particles | `state_space.py`, `state_space_nonlinear.py`, `state_space_estimation.py` |
| bayesian | Conjugate updates/predictives, total variance, NUTS/VI comparison, sequential evidence | `bayesian.py`, `bayesian_pymc.py`, `bayesian_sequential.py` |
| regimes | Markov/HMM recursions, switching recovery, Poisson/Hawkes likelihoods and compensators | `regimes.py`, `events.py` |
| processes | Generators, OU semigroups/stationarity, CIR moments, jumps, deterministic limits and coupled SDE convergence | `processes.py`, `processes_backends.py` |
| rough | Fractional scaling, Brownian limits, cross-covariance, tower property and Bergomi reductions | `rough.py` |
| simulation | Joint Gaussian paths, Brownian increments, coupled Euler/Milstein convergence, antithetic/QMC error semantics | `simulation.py` |
| measures | GBM change of measure, numeraire identities, drift-span rejection and reweighting | `measures.py` |
| contracts | Settlement, schedules, cashflows, interest factors and corporate actions | `contracts.py` |
| curves | Discount/zero/forward identities, bootstrapping, carry and generated-data curve fitting | `curves.py` |
| pricing | Put-call parity, Greeks, American/European ordering, lattice/PDE convergence and scenario pricing | `pricing.py`, `financial_workflows.py` |
| implied_volatility | Price/volatility inversion, bounds, SVI/SSVI constraints, SABR/local-volatility checks | `implied_volatility.py` |
| calibration | Parameter recovery, Jacobian rank, identification and failed-solver reporting | `calibration.py` |
| stopping | Small known stopping problems, exercise logic and out-of-sample policy evaluation | `stopping.py` |
| sensitivities | Finite-difference comparisons, hedge linear identities and key-rate risk | `sensitivities.py` |
| fixed_income | Price/yield inversion, duration/convexity, floating fixings and callable bond ordering | `fixed_income.py` |
| rates | Curve-consistent short-rate pricing, process adapters, cap/swaption conventions and market-model paths | `rates.py` |
| credit | Survival/hazard identities, CDS bootstrap/repricing and migration conservation | `credit.py` |
| risk | Fractional-atom ES, translation/homogeneity/subadditivity and Euler allocations | `risk.py` |
| utility | Inverse utilities, Jensen/CE identities, negative wealth domains and stable exponential utility | `utility.py` |
| optimization | Known LP/QP/SOC optima, duality, integer enumeration and failure propagation | `optimization.py` |
| robust | Support functions, minimax/CVaR, KL limits and Gaussian chance constraints | `robust.py` |
| control | Bellman recursion, information timing, scalar Riccati identity, nonanticipation and HJB limits | `optimal_control.py` |
| policies | Feasibility masks, thresholds, tie rules and switching/no-action decisions | `policies.py` |
| execution | Cost identities, gradients and positive-semidefinite impact costs | `execution.py` |
| wealth | Self-financing trade accounting, funding and financed gains | `wealth.py` |
| diagnostics | Proper scores against direct formulas, dependence sensitivity, HAC and paired block bootstrap | `diagnostics.py` |
| reinforcement | Q-learning/fitted-Q against exact control, terminal masks, tensor gradients and real A2C/PPO | `reinforcement.py` |

## Cross-family results

The new `test_end_to_end.py` uses only public imports. It recovers log-return
parameters from generated observations, constructs a GBM P/Q pair explicitly,
simulates the Q process, converts complete paths into forecasts, checks the
stock/numeraire martingale, calculates tail risk, solves a projection hedge, and
checks an LQR Riccati solution using the estimated growth coefficient. Reference
answers come from identities, a directly sorted tail and a scalar analytic
optimum, rather than the same provider called twice.

`test_readiness_composition.py` additionally checks option Monte Carlo prices
against Black–Scholes, forecast/path covariance preservation, non-IID
reweighting error semantics and sensitivity-to-hedge composition.
`decision_workflows.py` carries a Q option premium into financed P wealth,
transaction costs, expected utility, ES constraints and finite action selection.
The financial workflow covers quotes → curves → instruments → pricing → risk.

Meaning preserved at these boundaries: scenario weights; complete-path temporal
dependence; horizon-major flattening; discounted payoff units; explicit P versus
Q selection; covariance versus standard deviation; and approximate versus exact
sampler labels. A finite empirical law is never certified as the original
continuous law, and marginal state/forecast intervals do not create a joint law.

## API, human and AI-agent usability

**What works:** family names resolve through `convexity.math`; common workflows
use arrays, Python dates/strings and callbacks. Export tables, examples and
family guides make ordinary entry points discoverable. Structural model,
distribution, process, solver, policy and sampler protocols allow extensions
without registry edits. Existing tests exercise these user-defined paths and
reject wrong shapes/results. Native provider objects remain available through
explicit advanced escape hatches; this is not a promise to wrap every vendor
feature.

**Inconsistencies that remain:** scalar process paths use `(draw,time)` while
vector states use `(draw,time,state)`; forecasts always use `(draw,horizon,target)`.
A scalar forecast marginal and a one-coordinate vector distribution are distinct
objects. Result access mixes methods (distribution moments) and properties
(state moments), while some legacy result aliases remain. These choices are
usable when documented but cannot safely be guessed by an agent from a name.

The updated inventory and generated reference describe public names, including
modules, protocols and aliases; these are not independent mathematical
capabilities. All exported SDK callables have authored documentation and public
signatures without explicit `Any`. Numerical array shapes and units remain
runtime/semantic contracts. Private provider dispatch, optional native objects
and some method internals still use inferred or dynamic types: passing basic
Pyright is not a strict-typing certificate. An agent can discover ordinary input
and result types from the public surface, but must still inspect capabilities
before composing optional operations. Point forecasts, conditional variances and
full joint predictive laws deliberately remain different contracts.

Curve-fit callers that used provider fields should migrate `.fun` to
`.diagnostics.residuals` and `.jac` to `.diagnostics.jacobian`; `.parameters` is
the preferred spelling of the retained `.x` alias. The reference date is now
validated and retained instead of being ignored.

Serialization is supported where an explicit artifact format or owned numerical
record exists. Arbitrary Python callbacks, provider models, native handles and
training checkpoints do not share a portable, version-independent serialization
schema. Save constructor inputs, fitted parameters, feature/state order, RNG
configuration and dependency/checkpoint versions. Pickle round trips in one
trusted environment are not a durable cross-version contract.

## Numerical, semantic and integration limitations

- Rank at the floating-point boundary remains numerical. Validation is now in
  dimensionless coordinates; small negative roundoff may be projected to zero.
  This is not interval arithmetic or a proof that arbitrarily ill-conditioned
  inputs are resolved. General linear algebra, optimizers and calibration still
  require suitable scaling and diagnosis of conditioning/identification.
- Joint/marginal forecast moment agreement cannot prove equality of arbitrary
  user-defined distributions. Supplying both is an assertion that they are the
  same law. The constructor checks available moments; it cannot verify arbitrary
  callbacks mathematically.
- Scenario arrays have no enforced probability-measure provenance. Supplying a
  Q `MeasureModel` to scenario pricing explicitly asserts that the supplied paths
  follow that law; it does not transform P paths into Q paths. The drift-span
  test is local and does not prove global Girsanov integrability. End-to-end tests
  construct/simulate the pricing law explicitly.
- Gaussian forecasts and volatility laws condition on their stated fitted
  coefficients and future covariates. They do not automatically include parameter
  uncertainty. Neural and pretrained execution checks establish functioning
  adapters, not universal predictive accuracy or calibrated uncertainty.
- Solver tolerances, integration grids, Monte Carlo errors and approximate
  inference remain substantive mathematical choices. Status/diagnostics and
  convergence studies must be consumed; finite tests do not certify global
  nonlinear optima, MCMC convergence or rare-event accuracy.
- Provider global state is isolated in tested wrappers, but arbitrary direct
  provider calls from another thread are outside those locks. Bitwise behavior
  across CPU/GPU platforms, BLAS versions or provider upgrades is not certified.
- The provider suite emits known warnings: intentional rank/convergence probes,
  sktime/PyTorch deprecations, Lightning worker hints and a NeuralForecast/Pandas
  read-only-array conversion warning. No mutation or incorrect output was
  demonstrated for that conversion, but it remains an upstream compatibility
  risk to track. Warnings were not globally suppressed.

## Coverage gaps and concrete release blockers

No known reproduced numerical defect is intentionally left as an expected
failure. For the requested Python 3.14 quant environment and tested bounded
catalog, there is no outstanding blocker from the public-contract work.
The original undocumented-entry-point and ordinary-signature `Any` findings
are addressed by code contracts, the generated reference and regression gates.
Other Python versions are outside this follow-up's requested scope.

The following limits remain relevant to serious use:

1. The original branch coverage is incomplete and was not remeasured for this
   typing/documentation follow-up. Arbitrary custom callbacks, model combinations
   and every numerical boundary cannot be established by the finite suite.
2. Private dynamic provider dispatch and advanced native configuration are not
   fully statically typed. Some class methods rely on inferred annotations.
   Shapes, physical units, distributional assumptions and measure provenance
   require the documented runtime checks and caller discipline.
3. Exact reproducibility across provider upgrades, hardware and concurrent
   direct native calls is not certified. The evidence applies to recorded quant
   dependency versions; package lower bounds are not a tested version matrix.
4. Extreme conditioning, approximate inference, parameter identification and
   uncertainty calibration need application-specific acceptance tests. A solver
   success flag or green integration test does not certify these properties.
5. Arbitrary callbacks and provider handles lack a universal portable persistence
   format. Save inputs, units, coordinate order, RNG/checkpoint identity and
   dependency versions with artifacts.

These are explicit boundaries on the readiness claim, not requests to broaden
the 36-family catalog. Production use outside them needs its own validation.
