# Mathematical implementation audit — 2026-09-30

Families 1–12 contain actual numerical implementations and library integrations.
Their evidence is uneven: many foundational algorithms have analytic-reference
tests, while several forecasting architectures have primarily integration tests.
The earlier passing suite was insufficient to establish the quality of every
advertised model. This audit found and corrected failures that it missed.

This is a bounded source review and execution audit, not a proof of every
algorithm over every input. No empirical trading-performance claim follows from
these checks.

## Environment and execution

All execution used `/home/ethan/miniconda3/envs/quant/bin/python`, Python 3.14.7.
The environment includes the optional numerical, neural, forecasting,
state-space, Bayesian and process dependencies. PyTorch 2.14.0 uses its CUDA
13.0 runtime on the RTX 4060 Laptop GPU; the machine's toolkit is CUDA 13.1.
`python -m pip check` reports no broken requirements.

All **20 example scripts for families 1–12** completed, including numerical
autodiff, actual PyMC inference, particle filtering, all five cached foundation
checkpoints and one-step Chronos-2 fine-tuning. Examples for distributions,
estimation, GPs, state-space, conjugate Bayes and advanced forecasting were
rerun after the changes. The checkpoint revisions are recorded in
[forecasting completion](forecasting_completion.md#actual-checkpoint-verification).

The full suite completed with **1,321 passed, zero failed and zero skipped**
in 271.48 seconds, including 123 forecasting tests, 61 process tests and 45 new
cross-family covariance checks. The audit added 57 test cases overall.
The final artifact checks were also rerun separately: **13 passed**.
Whole-repository Pyright finished with **zero errors and zero warnings** using
the quant interpreter.
Five additional CUDA runs exercised training and path generation for flow, VAE,
autoregressive, score and diffusion models. All returned finite paths and
preserved the caller's global CUDA random state. These short GPU runs establish
execution and RNG isolation, not accuracy.

The suite emitted 70 warnings, retained in the log: deliberate failed-fit/rank
tests, a SciPy ill-conditioning diagnostic, sktime/Lightning deprecations and
CPU/data-loader notices. NeuralForecast also warns when wrapping a read-only
NumPy array; inspection of its installed `_as_torch_copy` implementation confirms
it immediately clones the tensor before returning it. These warnings were not
suppressed or counted as skipped tests.

## Evidence by family

The table identifies substantive existing checks inspected during the audit.
Backend agreement verifies Convexity's mapping to that backend; analytic
identities and exhaustive references provide additional independent evidence.
Neither a large test count nor one synthetic recovery experiment certifies
general accuracy.

| Family | Evidence beyond output shapes | Boundary of the evidence |
|---|---|---|
| 1. Numerical mathematics | Known integrals and derivatives; linear-system residuals; eigen/SVD reconstruction; ODE/BVP analytic solutions; finite-difference refinement; sparse operations without densification; Lyapunov/Riccati residuals | Tested methods and tolerances, not universal convergence or conditioning guarantees |
| 2. Distributions | Density/CDF/quantile identities; conditional Gaussian/Student-t laws; mixture reweighting; change of variables; empirical moments and support | Declared capabilities only; new tests reject covariance asymmetry at small scales |
| 3. Estimation | Normal equations; constrained feasibility; likelihood recovery; independent SciPy bootstrap comparisons; delta-method identities; failed-fit handling | Local optimizer convergence is not a global-optimum guarantee; covariance validation corrected |
| 4. Transforms and geometry | PCA reconstruction and optimal residual error; whitening covariance; factor posterior against Gaussian conditioning; differencing round trips; projection identities; causal lags | Reconstruction and training-only preprocessing under the declared rank assumptions |
| 5. Dependence and tails | Covariance/shrinkage references; copula uniform margins and elliptical-law recovery; Kendall inversion; GPD/GEV parameter recovery and tail limits | Finite generated-data experiments do not establish tail-risk reliability on market data |
| 6. Predictive learning | OLS/ridge/elastic-net closed forms; GLM frequency replication; quantile optima; tree recovery; neural nonlinear recovery on held-out inputs; generated logistic probabilities | Real fitting and controlled recovery, not broad generalization or model selection |
| 7. Gaussian processes | Posterior and marginal likelihood against Gaussian conditioning; finite-difference kernel gradients; interpolation; correlated joint sampling | Scalar-response GP catalog; numerical conditioning and kernel suitability remain input dependent; covariance validation corrected |
| 8. Forecasting | AR/VAR analytic forecasts and joint covariance; exact regression dynamics; no-future-data tests; neural periodic recovery; flow inverse/Jacobian identities; new recovery tests for all five native generators | Dedicated NeuralForecast architectures and foundation checkpoints mainly have execution/contract evidence. No broad accuracy or calibration benchmark. Diffusion recovery failure corrected |
| 9. Volatility | GARCH coefficient recovery and multistep recursion; pathwise ARCH/GJR/EGARCH/EWMA recursions; Monte Carlo versus analytic variance; return-unit scaling; HAR Jensen correction and joint covariance | Generated-data and backend references; no empirical market-volatility forecasting claim |
| 10. State-space | Dense Gaussian conditioning with missing data; EKF/UKF linear reduction; exact UKF moments; particle inference against Kalman and finite-state enumeration; elapsed-time semigroup | EKF/UKF remain approximations; finite-particle error remains; covariance validation corrected |
| 11. Bayesian inference | Exact conjugate updates; regression against Gaussian conditioning; NUTS/VI against conjugate references; sequential evidence against exact prefixes | MCMC diagnostics do not prove convergence; VI is approximate; static-parameter particles can lose diversity; prior covariance validation corrected |
| 12. Regimes and events | Forward/backward/Viterbi against exhaustive enumeration; AR(2) regime history enumeration; switching recovery; Poisson time change; Hawkes compensator/likelihood; compound-event moments | Synthetic and finite-state references; no claim of identifying real-world regimes or catalysts |

Relevant tests live under `tests/math/<module>/`. Family 13 was audited in the
preceding process implementation and remains included in the full regression run.

## Findings and corrections

### Covariance validity depended on measurement units

Distributions, estimation, GPs, state-space and Bayesian regression used absolute
symmetry or positive-semidefinite tolerances that could dominate small matrices.
For example, `1e-20 * [[1,2],[2,1]]` has a negative eigenvalue equal in magnitude
to its diagonal, but GP/state-space constructors accepted and projected it;
the estimation result also accepted it. Several constructors accepted
`1e-20 * [[2,.6],[.2,1]]`, although it is materially asymmetric.

Checks now scale with the matrix itself. Bayesian covariance matrices also
become exactly symmetric before Cholesky factorization. No unit-sized jitter
is introduced. The new
[unit-invariance tests](../../tests/math/test_covariance_units.py) cover six public
constructors at covariance scales `1e-20`, `1`, and `1e20`: valid matrices retain
their units; materially asymmetric or indefinite inputs are rejected.
Before the fix, **9 of these 45 cases failed**.

### A diffusion model could pass its tests while failing elementary recovery

The old diffusion test checked finite paths, reproducibility, nonzero variance
and save/load behavior after four training epochs. Those checks did not require
the model to learn a useful conditional forecast.

A new experiment trains on 50 repetitions of `[0,1,0,-1]` and forecasts the
next four observations. All five native generators use context/horizon 4,
hidden size 32, 100 epochs, seed 123, and 2,000 sampled paths with RNG seed 42.
No held-out target is supplied to fitting. This is intentionally an elementary
conditional recovery problem, not a diverse forecasting benchmark.

| Generator | Forecast-mean RMSE | Largest forecast standard deviation |
|---|---:|---:|
| Flow | 0.00329 | 0.0171 |
| VAE | 0.00732 | 0.0194 |
| Autoregressive | 0.00156 | 0.1217 |
| Score | 0.04610 | 0.3720 |
| Diffusion, before correction | **8.87775** | **3.2751** |
| Diffusion, after correction | **0.05729** | **0.3372** |

Epsilon prediction required dividing errors by the very small terminal signal
of the cosine schedule. New training uses velocity prediction, reconstructing
the clean path without that division, and the exact DDPM posterior-mean
coefficients. This is a standard parameterization described in
[Salimans and Ho](https://arxiv.org/abs/2202.00512) and implemented by
[Diffusers' DDPM scheduler](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddpm.py).
The reported comparison holds data, seed, network size and training budget fixed.

The new [accuracy tests](../../tests/math/forecasting/test_generative_accuracy.py)
require all five generators to recover the held-out cycle with mean RMSE below
0.15 and path RMSE below 0.5. Analytic known-path denoisers independently verify
the sampler for 2, 7 and 50 diffusion steps, for both epsilon and velocity.

Old pickle and portable epsilon models retain their original interpretation.
New portable models use the explicit `ddpm-velocity` engine kind, so older
readers reject them instead of silently treating their weights as epsilon.
[Artifact tests](../../tests/math/forecasting/test_artifacts.py) cover both formats.

**Remaining limitation:** score and diffusion samples still have excess spread
for this deterministic target. Their means can recover this pattern without
their uncertainty being calibrated. A fixed training budget is not convergence,
and these experiments do not demonstrate skill or calibration on another
dataset. The specialist neural architectures and pretrained models also need
dataset-specific temporal evaluation before being relied on for predictions.

## Reproducing the audit

```bash
conda activate quant
python -m pip check
python -m pytest -q -rs --junitxml=.cache/family-audit/full-suite.xml
python -m pyright --pythonpath "$(command -v python)"
python -m pytest tests/math/test_covariance_units.py tests/math/forecasting/test_generative_accuracy.py -q
python docs/math/examples/numerical_mathematics.py --autodiff
python docs/math/examples/forecasting_pretrained.py
```

Run the remaining scripts in `docs/math/examples/` with the same interpreter. The local
audit cache contains example exit codes/logs, before/after generative measurements,
GPU checks, JUnit results and type-check output. These cache artifacts are not
source-controlled; the substantive new regression tests and this report are.
