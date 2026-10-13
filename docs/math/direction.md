# Mathematical framework roadmap

The current repository implements families **1–36** under `convexity.math`.
Families **16–26** have a [bounded financial catalog, contracts and verification](financial_mathematics.md),
with QuantLib as a principal dependency. The [financial QA review](financial_qa.md)
records architecture corrections, adversarial regressions and executed notebooks.
The [second QA and public API readiness pass](api_readiness.md) checks the
composition contracts needed by the remaining families.
Families **27–36** have a [bounded risk and decision catalog](decision_mathematics.md),
including array-based optimization, finite and belief-state control, execution,
wealth transformations, diagnostics and optional reinforcement learning.
Each family has executable examples, mathematical tests and executed notebooks.
The catalogs, rather than every possible model in these broad subjects, define completion.

The [2026-09-30 mathematical audit](mathematical_audit.md) records the evidence
and limitations for families 1–12, including defects missed by the earlier
suite. Implementation completion does not establish predictive accuracy or
calibrated uncertainty for every model or dataset.

This repository contains generic mathematical tools, their tests and generated-data
examples. Trading runtimes, broker connectivity and market-data acquisition
belong to later systems built with Convexity. The
[mathematical trading-system specifications](mathematical_trading_systems)
remain reference requirements; the catalog extends beyond those sheets to bonds,
Gaussian processes and broader forecasting and neural models. RL remains optional.

**1–7: Numerical, probability and statistical foundations**

| # | Mathematical family | Objects and methods to support | Implementation approach |
|---|---|---|---|
| **1** | **Numerical mathematics** | Dense/sparse and matrix-free linear algebra, least squares, generalized eigenproblems, factorizations, matrix functions, Lyapunov/Riccati equations; roots/fixed points; derivatives, JVP/VJP and derivative checks; quadrature; interpolation/approximation; ODE IVPs/BVPs; finite-difference operators, grids, and accuracy diagnostics. | Implemented with NumPy/SciPy, optional PyTorch autodiff, and findiff operators. See the [completed numerical catalog and contracts](numerical_mathematics.md). |
| **2** | **Probability distributions and conditional laws** | Scalar/vector distributions; Gaussian, Student-t, lognormal, Bernoulli, binomial, Poisson; empirical distributions, mixtures and transformed distributions. Density, log density, CDF, quantiles, sampling, moments, conditioning and expectations where supported. | Implemented with NumPy/SciPy and optional PyTorch tensor laws. See the [completed distribution catalog and contracts](distributions.md). |
| **3** | **Statistical estimation** | Estimators, likelihoods, parameter constraints, least squares, maximum likelihood, MAP, regularization, robust estimation, confidence intervals and parameter uncertainty. | Implemented with SciPy optimization/resampling, statsmodels and scikit-learn. See the [completed estimation catalog and contracts](statistical_estimation.md). |
| **4** | **Transforms, decomposition and factor geometry** | Scaling, differencing, lag construction, seasonal/Fourier features, PCA, whitening, factor models, projections, residuals, inverse transforms and subspace alignment. | Implemented with scikit-learn, statsmodels and SciPy. See the [completed transforms catalog and contracts](transforms.md). |
| **5** | **Covariance, dependence and tails** | Sample/EWMA covariance, shrinkage, factor covariance, correlation, positive-semidefinite constraints, multivariate heavy tails, copulas and extreme-value tail models. | Implemented with scikit-learn covariance estimators, SciPy, statsmodels copulas and the completed distribution family. See the [completed dependence catalog and contracts](dependence.md). |
| **6** | **Regression and general predictive learning** | Linear/ridge/lasso/elastic-net regression, GLMs, logistic/binomial models, quantile regression, trees, ensembles and neural functions. Conditional mean, probability and quantile predictions. | Implemented with scikit-learn, statsmodels and optional PyTorch. See the [completed predictive-learning catalog and contracts](predictive_learning.md). |
| **7** | **Gaussian processes and kernels** | Kernel functions, kernel composition, GP regression, posterior mean/covariance, predictive distributions and hyperparameter fitting. A broader research capability, not a requirement of the baseline sheets. | Implemented with scikit-learn GP regression and kernels, with SciPy optimization and joint Gaussian predictions. See the [completed Gaussian-process catalog and contracts](gaussian_processes.md). |

SciPy already supplies the numerical foundation; scikit-learn supplies substantial estimation, decomposition and learning functionality. We should integrate those implementations. [SciPy API](https://docs.scipy.org/doc/scipy/reference/index.html), [scikit-learn API](https://scikit-learn.org/stable/api/index.html), [Gaussian processes](https://scikit-learn.org/stable/modules/gaussian_process.html).

**8–15: Forecasting, inference and stochastic dynamics**

| # | Mathematical family | Objects and methods to support | Implementation approach |
|---|---|---|---|
| **8** | **Time-series forecasting** | Forecasting contracts, horizons, covariates and probabilistic results; baselines; classical and multivariate models; regression, dedicated neural architectures and generative forecasting; real foundation checkpoints and fine-tuning; elapsed-time and shared panel forecasting; ensembles, temporal stacking, backtesting, calibration, diagnostics, selection and reproducible artifacts. | Implemented with statsmodels, scikit-learn, optional PyTorch, sktime, NeuralForecast and real pretrained providers. See the [forecasting catalog and contracts](time_series_forecasting.md), and the [expanded completion record](forecasting_completion.md). |
| **9** | **Volatility forecasting** | Conditional variance and volatility forecasts; constant/EWMA baselines, ARCH, GARCH, GJR-GARCH, EGARCH, realized variance and level/log HAR; four standardized innovation laws, multi-step analytic/simulation/bootstrap forecasts, variance backtests and diagnostics. Distinct from implied-volatility surfaces. | Implemented through the existing forecasting and distribution contracts with `arch` and statsmodels. See the [completed volatility catalog and contracts](volatility_forecasting.md). |
| **10** | **State-space models, filtering and smoothing** | Transition/observation kernels and latent-state laws; Kalman, extended and unscented filters and smoothers; bootstrap particle filtering and backward/ancestral smoothing; missing observations; exact elapsed linear dynamics and explicit irregular-time callbacks. | Implemented through existing TimeSeries and Distribution contracts with statsmodels, FilterPy, particles and SciPy. See the [completed state-space catalog and contracts](state_space.md). |
| **11** | **Bayesian inference and parameter uncertainty** | Priors, likelihoods, posterior distributions, conjugate updates, posterior prediction, MCMC, variational inference and joint state/parameter inference. | Implemented with exact NumPy/SciPy updates, PyMC MCMC/ADVI and joint static-parameter/state bootstrap SMC through particles. See the [completed Bayesian catalog and contracts](bayesian_inference.md). |
| **12** | **Regimes, catalysts and event models** | Markov chains, hidden regimes, switching dynamics, scheduled event-surprise distributions, jump-size distributions and event intensities. | Implemented with statsmodels switching regression/AR, exact finite-state inference, existing distributions and composed scheduled/Poisson/Hawkes event laws. See the [completed regimes and events catalog](regimes_events.md). |
| **13** | **Stochastic processes and differential equations** | Transition kernels, drift/diffusion functions, generators and exact/discretized transitions. Brownian motion, arithmetic/geometric Brownian motion, scalar/vector OU, CIR, Heston and jump diffusions. | Implemented with exact NumPy/SciPy laws, existing linear state-space transitions, QuantLib Heston QE, and sdeint custom Itô integration. See the [completed stochastic-process catalog and contracts](stochastic_processes.md). |
| **14** | **History-dependent and rough processes** | Fractional/Volterra Gaussian drivers, memory kernels, forward-variance states, rough Bergomi, and finite-factor approximations with approximation-error checks. Covered in the broader options volume. | Implemented with specialized NumPy/SciPy Gaussian and kernel calculations, shared Brownian/OU factors, explicit history conditioning, and integrated kernel-error checks. See the [completed rough-process catalog and contracts](rough_processes.md). |
| **15** | **Simulation and joint path distributions** | Correlated innovations, weighted scenarios, joint paths across assets/time, exact sampling, Euler/Milstein and model-specific schemes, Monte Carlo/quasi-Monte Carlo, variance reduction and simulated observations. | Implemented with NumPy/SciPy joint laws and drivers, scalar Milstein, existing process kernels including QuantLib/sdeint, and explicit sampling-design error estimates. See the [completed simulation catalog and contracts](simulation.md). |

statsmodels already covers a substantial classical forecasting and state-space catalog. `arch` supplies volatility models and statistical tools. PyMC and `particles` address different forms of inference; they are not interchangeable. [statsmodels](https://www.statsmodels.org/stable/tsa.html), [sktime forecasting](https://www.sktime.net/docs/api-reference/forecasting/), [arch](https://bashtage.github.io/arch/), [PyMC](https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/pymc_overview.html), [particles](https://particles-sequential-monte-carlo-in-python.readthedocs.io/en/latest/). Diffrax is a candidate backend for custom ODE/SDE dynamics. [Diffrax](https://docs.kidger.site/diffrax/).

**The forecasting family is a complete, bounded project.** Its [catalog](time_series_forecasting.md) supplies:

- A public interface for user-written forecasters.
- A declared catalog of working model adapters.
- Univariate/multivariate inputs, horizons and exogenous-variable conventions.
- Explicit treatment of missing values, frequency and array dimensions.
- Point, quantile, interval, distribution and path outputs **according to each model’s actual capabilities**.
- Fit/update/predict behavior, fitted parameters and diagnostics.
- Numerical tests and generated-series examples.

A point forecast must not be presented as a probability distribution by silently adding Gaussian uncertainty. Likewise, separate interval forecasts do not automatically define a joint path law.

**16–23: Financial valuation and options mathematics**

| # | Mathematical family | Objects and methods to support | Implementation approach |
|---|---|---|---|
| **16** | **Physical and pricing measures: P and Q** | Physical laws, pricing laws, numeraires, discounting, likelihood-ratio reweighting, market prices of risk and model-specific measure changes. Paired P/Q models with compatible diffusion structure. | QuantLib pricing models plus explicit Convexity measure/model composition. There is no universal “convert any P forecast into Q” operation. |
| **17** | **Payoffs, cash flows and contract mathematics** | Payoff functions, coupon/redemption schedules, dividends, multipliers, exercise styles, cash versus physical settlement, currency conversion, day counts and compounding conventions. | QuantLib instruments, schedules and cash-flow primitives; small custom payoff functions. |
| **18** | **Curves, forwards, basis and carry** | Discount factors, zero/forward rates, dividend and borrowing curves, forward prices, futures basis, storage/convenience-yield models, curve interpolation and bootstrapping. | QuantLib curves and cash-flow calculations; SciPy for additional fitting. |
| **19** | **Derivative pricing engines** | Pricing problems and pricing results. Black–Scholes–Merton, Black-76, Bachelier, Heston; analytic/transform methods, trees, finite differences and Monte Carlo. European, path-dependent and multi-asset claims as separately declared capabilities. | QuantLib first. Add bespoke algorithms only for genuine gaps. |
| **20** | **Implied volatility and volatility surfaces** | Implied-volatility inversion, smiles, total-variance surfaces, SVI/SSVI, interpolation, static-arbitrage constraints, risk-neutral densities and Dupire local volatility. SABR is an additional useful family, especially for rates. | QuantLib where supported; SciPy fitting; custom constraints where the required surface specification is not provided. |
| **21** | **Calibration and inverse problems** | Calibration objectives, quote weights, parameter bounds, regularization, Jacobians, identifiability, uncertainty and fit diagnostics. Recover model parameters from synthetic prices or supplied observations. | QuantLib calibration engines and SciPy optimizers. |
| **22** | **Optimal stopping and early exercise** | Stopping problems, Snell envelopes, continuation values and exercise regions; American/Bermudan pricing; dividend events; backward induction, obstacle PDEs and least-squares Monte Carlo. | QuantLib engines plus explicit model/payoff interfaces. |
| **23** | **Sensitivities, exposures and hedging geometry** | Delta, gamma, vega, theta, rho, cross-Greeks; curve sensitivities; Jacobians/Hessians; hedge ratios, exposure matrices, nullspaces and constrained hedging. | Analytic/QuantLib Greeks, finite differences and automatic differentiation where supported. |

QuantLib should be a principal dependency for financial mathematics. It already separates instruments from pricing engines and provides extensive financial infrastructure. Convexity’s role is to make those capabilities usable alongside our forecasts and model interfaces. [QuantLib documentation](https://www.quantlib.org/docs.shtml). Automatic differentiation should also come from an established engine. [PyTorch autograd](https://docs.pytorch.org/docs/2.14/autograd.html).

**24–26: Additional families for bonds and rates**

These extend the scope beyond the futures/SPY specifications.

| # | Mathematical family | Objects and methods to support | Implementation approach |
|---|---|---|---|
| **24** | **Bond and fixed-income analytics** | Zero-coupon, fixed/floating-rate, amortizing and inflation-linked cash flows; accrued interest; clean/dirty prices; yield-to-maturity; duration, convexity, DV01, key-rate duration, carry and roll-down. Callable/putable bonds reuse optimal stopping. | QuantLib bond and cash-flow analytics. |
| **25** | **Interest-rate and inflation dynamics** | Short-rate and term-structure models: Vasicek, CIR, Hull–White and multifactor Gaussian models. Broader extensions: HJM/market models, inflation dynamics, caps/floors and swaptions. Separate P forecasting from Q valuation. | QuantLib rate models and engines; forecasting adapters for empirical curve dynamics. |
| **26** | **Credit and default mathematics** | Survival probabilities, hazard curves, recovery, default-time distributions, risky cash flows, credit spreads, spread sensitivities and migration matrices. Structural versus reduced-form models. CDS can serve as a calibration instrument. | QuantLib credit curves/pricing plus statistical estimation adapters. |

QuantLib’s maintainer-authored guide covers cash flows, bond curves, curve bootstrapping, bond analytics, inflation and default-probability curves. These are strong candidates for direct integration rather than reconstruction. [A QuantLib Guide](https://www.quantlibguide.com/).

For **spot/equities**, most mathematics is already covered by the shared families: total returns, dividends, financing/borrowing, factors, forecasts, risk and allocation. Corporate actions require explicit transformations of holdings and prices. They do not require an entirely separate forecasting framework.

**27–36: Risk, decisions and the remaining mathematics in the sheets**

These can also be developed as standalone mathematical tools. Their inputs can be supplied arrays, functions and generated scenarios.

| # | Mathematical family | Objects and methods to support | Implementation approach |
|---|---|---|---|
| **27** | **Risk measures and stress mathematics** | Variance, downside measures, VaR, ES/CVaR, drawdown, barrier/breach probabilities, marginal/component risk, stress transformations and scenario-based margin functions. | SciPy/NumPy and optimization backends; explicit definitions for distributional versus path risk. |
| **28** | **Utility and decision criteria** | Expected utility, mean–variance, mean–CVaR, log/exponential/power utilities, certainty equivalents and the options sheet’s specified utility. Turnover and switching preferences. | Small mathematical functions evaluated analytically or over weighted scenarios. |
| **29** | **Mathematical optimization** | Variables, domains, objectives, constraints and solve results. LP, QP, conic, nonlinear, mixed-integer and bounded finite problems; feasibility, dual variables, optimality gaps and solver diagnostics. | CVXPY with compatible solvers; SciPy for nonlinear problems; enumeration for small exact references. |
| **30** | **Robust and stochastic optimization** | Uncertainty sets, support functions, ellipsoidal mean ambiguity, minimax objectives, scenario approximations, chance constraints and distributionally robust extensions. | CVXPY/SciPy plus explicit mathematical formulations. |
| **31** | **Dynamic programming and optimal control** | States, actions, transition laws, rewards, value functions and policies. Bellman recursion, finite MDPs, belief-state problems, LQR/Riccati solutions, stochastic MPC, nonanticipative scenario trees and impulse-control/HJB formulations. | Established control/numerical libraries for supported cases; custom financial problem formulations. |
| **32** | **Threshold, hysteresis and stopping policies** | Probability thresholds, activation/exit bands, absorbing states, fixed deadlines and no-action regions. This is the mathematical part of the sheets’ thesis lifecycle. | Small deterministic policy/state-transition functions with exhaustive tests. |
| **33** | **Transaction costs, execution and assignment models** | Linear/quadratic impact, stochastic fill laws, optimal liquidation schedules, inventory-risk penalties and binomial/logistic assignment models. These are mathematical models—not broker connectivity. | SciPy/statsmodels estimation, CVXPY scheduling and supplied probability laws. |
| **34** | **Financed gains and wealth transformations** | Maps from holdings, outcomes and cash flows to wealth; self-financing identities; collateral transfers; lending/borrowing; coupons/dividends; exercise/delivery; exact versus approximate P&L attribution. | Explicit contract-specific equations, with QuantLib cash-flow support where useful. |
| **35** | **Statistical diagnostics and model comparison** | Residual tests, stationarity/cointegration tests, likelihood scores, CRPS, quantile scores, PIT/coverage, multivariate/path scores, bootstrap uncertainty, HAC covariance and multiple-comparison corrections. | statsmodels, `arch`, SciPy and scoring functions. All can be verified using generated samples. |
| **36** | **Reinforcement learning — optional** | Value-function approximation, Bellman losses, policy gradients, actor–critic methods, constrained actions and finite-state reference problems. | PyTorch and existing RL libraries; specialized constraint handling remains explicit. |

CVXPY should handle supported optimization formulations through appropriate numerical solvers; different solvers support different combinations of integer, quadratic and conic structure. [CVXPY solver capabilities](https://www.cvxpy.org/tutorial/solvers/index.html). `python-control` supplies established control tools, including LQR and related estimation/control methods. [Python Control](https://python-control.readthedocs.io/en/latest/). Existing RL implementations are available through Stable-Baselines3, although they do not automatically enforce the financial constraints in our sheets. [Supported algorithms](https://stable-baselines3.readthedocs.io/en/master/guide/algos.html).

**What “finish one family” should mean**

We should give each family a finite acceptance list before implementing it. “All forecasting” or “all optimization” is unbounded; a specified model catalog is finishable.

For each family, completion means:

1. **Usable mathematical interface:** documented input shapes, units, assumptions and outputs. Ordinary supported workflows must use public Convexity functions with arrays, dates, numbers and callbacks; provider imports, handles, settings and construction syntax belong only to optional advanced extensions.
2. **Working implementations:** the declared models actually run, including the selected library integrations.
3. **User extensibility:** a user can supply a compatible model without changing Convexity internals.
4. **Mathematical verification:** analytic identities, known solutions, generated-data recovery, numerical convergence and edge cases as appropriate.
5. **A small executable example:** supplied or generated inputs, inspectable results, no broker connection.

**Integrating a tested statsmodels forecaster or QuantLib pricing engine counts as implementation.** A class with empty methods does not. Convexity should provide consistent mathematical meaning, inputs, outputs and composition while the established libraries perform the numerical work.

No market-data acquisition or trading deployment is required to complete these mathematical families.
