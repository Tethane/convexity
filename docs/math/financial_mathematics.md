# Financial mathematics: families 16–26

These eleven families compose the existing mathematical core with **QuantLib
1.43 or later**, now a principal dependency. They follow the public
`interfaces/`, `builtins/` and `_backends/` organization: contracts and results,
public model configuration, and private vendor execution respectively. Native QuantLib objects remain
available where financial conventions belong to the backend. SciPy handles
inverse problems and constrained hedges. Sensitivities reuse the numerical
family, including its optional PyTorch autodiff implementation.

The [public API readiness guide](api_readiness.md) shows direct financial workflows,
readable conventions, owned fixing histories and composition with the core.

The [engineering QA record](financial_qa.md) documents reproduced defects, fixes,
architecture checks and remaining limits. Eleven [executed notebooks](notebooks/README.md)
provide generated-data investigations alongside the scripts.

## Shared contracts

- Rates, yields, spreads and lognormal volatilities are decimals. Times are
  years under the specified day counter. Normal volatility is price units per
  square root year. Greeks are per **unit**, not per percentage point.
- Dates accept ISO strings, Python dates or QuantLib dates. Day counts include
  Actual/365 Fixed, Actual/360, Actual/Actual ISDA and 30/360 Bond Basis, or a
  supplied QuantLib day counter. Schedule calendars, adjustment rules, stub
  dates, end-of-month behavior, and coupon frequency are explicit.
- Discount, dividend and volatility reference dates and process day-count clocks must agree. Pricing uses
  the supplied valuation date, never today's date implicitly. The shared
  `convexity.math.contracts.evaluation_date` context locks Convexity's calls and
  restores all QuantLib settings, including nested exceptions. External code
  using QuantLib global settings must cooperate with this context.
- Bootstraps are calibrated snapshots. Rebuild them after changing quotes.
  Native helper diagnostics must be evaluated inside their valuation-date
  context because relative-date helpers observe global QuantLib settings.
- Numerical arrays in result dataclasses are owned and read-only. Native
  QuantLib objects are mutable backend objects; an engine adapter attaches its
  engine to the supplied native instrument. They are not immutable snapshots.
- Unsupported engine/claim combinations fail. An unavailable Greek or Monte
  Carlo error remains absent/`None`, never zero. Curves do not extrapolate
  unless requested; flat curves have their native unlimited horizon.

## 16. Physical and pricing measures (`measures`)

**Acceptance catalog:** `MeasureModel`, `Numeraire`, money-market and deterministic
curve numeraires; compatible `DiffusionPair` models for GBM, OU/Vasicek, CIR and
Heston; market prices of Brownian risk; stochastic-exponential likelihood
ratios; stable scenario reweighting; numeraire-change density ratios.

`dW_Q = dW_P + lambda dt`, so `sigma lambda = b_P-b_Q` and the log density is
`-sum(lambda*dW_P) - 0.5*sum(||lambda||² dt)`. Brownian noise coordinates are
independent; correlation belongs in the diffusion matrix. State-dependent
`lambda` must be predictable. Its discretized stochastic exponential is not a
proof of integrability or equivalence over arbitrary horizons. Constant risk
uses the exact Gaussian likelihood ratio. P/Q pairs preserve diffusion
coefficients; drift changes outside their range are rejected.

`reweight(PathScenarios, log_likelihood)` returns a self-normalized finite law
with sampling-error information removed. It does not infer Q from an empirical
forecast. `MeasureModel.value` computes `N(0) E_Q[X/N(T)]` using an explicitly
supplied expectation. `BlackScholesModel.from_measure_pair` connects an explicit
GBM pair to QuantLib. [Example](examples/measures.py).

## 17. Payoffs, cash flows and contracts (`contracts`)

**Acceptance catalog:** vanilla and cash-digital payoffs; European, American and
Bermudan exercise; native schedules with stubs; fixed coupon/redemption legs;
explicit cash dividends; cash/physical delivery; currency conversion of cash
flows; day counts; continuous/simple/periodic compounding; share splits and
ex-dividend transformations.

A `VanillaContract` multiplier applies to price and Greeks. `delivery(spot)`
returns `(underlying units, cash)` and preserves exercise value under either
settlement convention. Currency labels alone do not create FX risk models.
`present_value` accepts supplied spot conversion factors in valuation-currency
units per source-currency unit; future FX-dependent claims need a joint model.
Reference-date cash flows are excluded by default. [Example](examples/contracts.py).

## 18. Curves, forwards, basis and carry (`curves`)

**Acceptance catalog:** flat, log-linear discount and linear-zero curves;
continuous zero/forward rates; parallel shifts; implied rolled curves; native
rate-helper bootstrapping with log-discount, linear-zero or flat-forward
interpolation; supplied-model SciPy curve fitting; equity/dividend/borrowing
carry; storage and convenience yield; futures-minus-forward basis.

`forward_price` computes a deterministic cash-and-carry forward. Discrete
dividends reduce prepaid spot; continuous yields reduce the forward. Borrowing
and storage are additional positive costs. Discrete and continuous dividend
specifications are mutually exclusive in this helper. Futures convexity under
stochastic rates is not silently set to zero: `futures_basis` accepts the
separately supplied futures price. `deposit_quote`, `swap_quote` and `ois_quote` construct calibration inputs from
numbers and tenors; `bootstrap` returns a curve with read-only
`calibration_errors`. Advanced callers can still supply native deposit, OIS,
swap, bond or futures helpers.
`fit_discount_curve` returns SciPy fit diagnostics for a supplied positive
parametric discount function. [Example](examples/curves.py).

## 19. Pricing engines (`pricing`)

`PricingProblem(contract, model, valuation_date)` requires a model explicitly
labeled Q. `price(problem, engine)` validates `PricingResult`, which contains
value, available Greeks, optional simulation error, engine name and currency.
Custom engines implement `price(problem)` without registration or inheritance.

| Implementation | Claims and exercise |
|---|---|
| `BlackScholesModel`, `VanillaEngine('analytic')` | European vanilla, BSM and analytic Greeks |
| `forward_option(..., model='black76'/'bachelier')` | European forward options, including shifted Black and negative normal forwards |
| `HestonModel`, `VanillaEngine('heston')` | European vanilla, QuantLib analytic transform integration |
| `VanillaEngine('tree')` | BSM CRR tree, European/American/Bermudan |
| `VanillaEngine('fd'/'heston_fd')` | BSM/Heston finite differences and early-exercise obstacle; BSM supports explicit discrete dividends |
| `VanillaEngine('mc'/'lsmc')` | European Monte Carlo / American least-squares Monte Carlo |
| `asian_price` | Discrete geometric analytic and arithmetic Monte Carlo average-price options; running sum/product for past fixings |
| `barrier_price` | Continuous single-barrier knock-in/out, analytic BSM |
| `basket_price` | Correlated European arithmetic weighted-basket Monte Carlo |
| `scenario_price` | Scalar path payoff on explicitly Q `PathScenarios`, deterministic numeraire |
| `QuantLibEngine(factory)` | Native instrument/engine extension; caller declares the native model's measure |

Tree, analytic barrier and analytic geometric-Asian engines require `flat_curve`
rates and constant volatility, because their native algorithms flatten the
coefficients. Use finite differences for shaped curves; a supplied volatility
surface uses QuantLib local-volatility dynamics in the BSM PDE. Invalid local
variance fails explicitly. One-step basket and fixing-grid arithmetic-Asian MC
require constant volatility. Basket models require the same native discount curve
or matching proven flat curves, not merely equal expiry discounts. MC seeds must
be positive; American MC uses separate fixed calibration and valuation seeds.
At constant zero volatility analytic prices remain available but Greeks are omitted.

Basket models require a common discount numeraire. `scenario_price` preserves
IID/antithetic/weighted sampling-error semantics from family 15. It can value
claims on supplied Q rough, jump or multi-asset paths; it does not re-label a P
forecast. [Example](examples/pricing.py).

## 20. Implied volatility and surfaces (`implied_volatility`)

**Acceptance catalog:** Black/shifted-Black/normal inversion with intrinsic and
upper-bound checks; raw SVI and constrained SVI fitting; power-law SSVI;
interpolated total-variance smile surfaces; QuantLib Black variance surfaces;
SABR normal/lognormal volatility; sampled butterfly/calendar diagnostics;
Breeden–Litzenberger density; QuantLib Dupire local volatility.

`SVI.fit` imposes positive minimum variance, Lee wing bounds and nonnegative
butterfly density on a declared constraint grid. `SmileSurface.diagnose`
checks every original expiry knot as well as intermediate expiries. These are **grid diagnostics**,
not a global arbitrage certificate. SSVI's `validate_domain(theta_max)` checks
sufficient bounds for its specified power-law family on that ATM variance
domain; ATM variance must be nondecreasing to obtain calendar consistency.
Smile interpolation rejects time extrapolation.

`black_variance_surface` inputs have `(strike, expiry)` axes. Local volatility
can fail on an inconsistent surface; negative variance is not clipped away.
Density finite differences expose negative values and depend on the chosen
step. [Example](examples/implied_volatility.py).

## 21. Calibration (`calibration`)

**Acceptance catalog:** callback-based inverse problems; weighted residuals;
bounds; quadratic prior regularization; SciPy least squares; native QuantLib
model/helper calibration; data-only Jacobians, singular values, numerical rank,
active bounds, convergence and local uncertainty diagnostics.

`CalibrationProblem.predict(parameters)` returns one scalar per observation.
The objective is `sum(weights * residual²) + regularization*||p-prior||²`.
Weights are nonnegative inverse-variance-style objective weights, not their
square roots. Bounds and prior share the parameter axis. The configurable rank
tolerance accounts for finite-difference noise. `covariance` is supplied only
for successful, locally identifiable, unconstrained, unregularized fits with
positive residual degrees of freedom; it is computed from the Jacobian SVD and is a local linear approximation,
not a posterior distribution. `quantlib_calibrate` uses the helper's native
error definition and mutates the supplied model. [Example](examples/calibration.py).

## 22. Optimal stopping (`stopping`)

**Acceptance catalog:** finite-state stopping problems; exact Snell envelopes,
continuation and exercise regions; Bermudan exercise masks; nonnegative-payoff
least-squares policies with independent-path evaluation; the QuantLib American,
Bermudan, dividend and obstacle-PDE engines from family 19; callable/putable
bond trees from family 24.

`StoppingProblem` uses rewards `(time,state)`, transition probabilities
`(time-1,state,state)`, and positive one-step discount factors (scalar, `(time-1,)` or
`(time-1,state)`). The terminal
reward is mandatory. `least_squares_policy` uses states `(draw,time[,state])`,
rewards `(draw,time)` and discounts broadcasting to `(draw,time-1)`.
`ExercisePolicy.evaluate` must use independent Q validation paths. It reports
a feasible-policy estimate and an IID standard error conditional on the trained
policy, not a dual bound or a training-bias correction. Path-dependent contracts
need sufficient running statistics in the state. [Example](examples/stopping.py).

## 23. Sensitivities and hedging (`sensitivities`)

**Acceptance catalog:** QuantLib Greeks; shared numerical Jacobians/Hessians;
central bump gradient, gamma and cross-Greeks; optional PyTorch autodiff for
tensor-native callbacks; curve derivatives; exposure matrices and nullspaces;
bounded/equality-constrained weighted least-squares hedges with optional ridge.

`hedge(A, target)` minimizes `||sqrt(W)(A h + target)||² + ridge||h||²` with
`A` axes `(risk,instrument)`. Its nullspace describes unconstrained exposure
neutrality; bounds and budgets may restrict feasible movements in it. Failed
or infeasible optimizations raise. `curve_sensitivities` reports derivatives
per unit decimal shift; positive loss DV01 is `-dV/dr * 1e-4`. QuantLib calls
are not differentiable by PyTorch; use their analytic Greeks or numerical
bumps. [Example](examples/sensitivities.py).

## 24. Bonds and fixed income (`fixed_income`)

**Acceptance catalog:** zero-coupon, fixed, floating, amortizing and CPI-linked
bonds; callable/putable fixed bonds; native cash flows; clean/dirty/accrued;
yield inversion; modified duration, convexity and DV01; triangular key-rate
duration; carry and roll-down decomposition.

Prices and yield sensitivities are per 100 current outstanding principal.
`price_from_yield` and `yield_from_price` accept an explicit `valuation_date`,
defaulting to the supplied settlement date. This controls fixing-versus-forecast
behavior independently of ambient QuantLib settings.
Floating coupons use the supplied index, forecast curve and required historical
fixings. Use `rates.rate_index(curve, fixings={date: rate})`; floating bonds
inherit that index’s fixing lag unless explicitly overridden. Supplied histories
are isolated to each valuation and restored afterward. CPI coupons retain QuantLib publication-lag/interpolation conventions.
Missing fixings fail instead of being invented. `compute_yield=False` avoids
reporting a yield when it is not useful for the instrument.

Yield duration is the sensitivity of promised cash flows to a single yield.
For callable bonds it is **not** effective option-adjusted duration. Callable
analytics require an explicit exercise engine. `key_rate_duration` requires
`engine_factory(curve)` for callable bonds and rebuilds the exercise model under
each curve bump. Callable carry is rejected because promised cash flows do not
represent an exercise-aware holding-period strategy.
`carry_roll_down` uses cash units, reinvested coupons and an explicit curve
evolution: constant remaining-tenor zero rates by default, or an implied
forward roll (zero excess return by construction). [Example](examples/fixed_income.py).

## 25. Rates and inflation (`rates`)

**Acceptance catalog:** QuantLib Vasicek, CIR, Hull–White and two-factor G2;
P OU/CIR forecasting transitions through family 13; Q discount bonds;
piecewise Gaussian HJM via native GSR; native forward market models with
QuantLib predictor-corrector simulation; empirical multivariate curve
forecasting; zero/YoY inflation curves and indices; deterministic-rate
lognormal CPI P/Q dynamics; caps/floors and European/Bermudan swaptions through
Black, normal, affine analytic, tree and G2 engines; inflation-swap valuation.

`ShortRateModel` validates named model parameters. Curve-fitted Hull–White/G2
are Q specifications. P Vasicek/CIR cannot price bonds. `ShortRateModel.process()` uses the shared Convexity process interface, including
Hull–White transitions and G2 Gaussian factor dynamics. G2 state is `(x,y)`;
its deterministic shift is supplied by the fitted model. QuantLib CIR valuation
requires its strict Feller domain `sigma² < 2*kappa*theta`; the existing core
CIR transition law also supports boundary-attainable P dynamics. Native GSR uses its
terminal forward measure. `forward_market_model` constructs a lognormal forward model directly from
rate times, forward levels, volatility and correlation parameters.
`market_model_paths` also accepts a native `MarketModel`
(e.g. `AbcdVol`), returns `(draw,evolution_step,forward_rate)` and weights, and
explicitly labels the terminal-forward numeraire. Neither output is silently
re-labeled as bank-account Q.

`EmpiricalCurveForecaster` composes family 8 with ordered tenors and retains
the existing fitted-forecaster result contract. Its empirical specification is
P. `InflationDynamics` uses a supplied P drift or deterministic nominal/real Q
curves, with growth `D_real/D_nominal`; it does not imply a stochastic-rate/CPI
correlation model. [Example](examples/rates.py).

## 26. Credit and default (`credit`)

**Acceptance catalog:** flat hazards and interpolated survival; CDS hazard
bootstraps; survival/default probability/density; censored default-time draws;
recovery-of-par risky cash-flow valuation; QuantLib CDS prices, fair spreads
and spread sensitivities; censored constant-intensity P estimation; absorbing
continuous-time rating migration and estimation; Merton structural debt/equity
and P/Q terminal-default probabilities.

`cds_quote` constructs forward-scheduled calibration instruments from decimal
spreads and tenors. `bootstrap_cds` exposes snapshot `calibration_errors`.
Use matching explicit trade dates when repricing calibration quotes.

`cds_price` accepts optional `trade_date`, `protection_start` and
`cash_settlement_days`; native schedule/protection-start compatibility still
applies. `CDSResult.premium_leg` is the net premium leg including
`accrual_rebate`, so it plus `protection_leg` equals `value`.
`spread_sensitivity` is per unit decimal contractual spread and includes rebate
exposure. The typed result also supports keyed access.

`HazardCurve` carries P or Q explicitly. Q is required for credit valuation.
`sample(..., horizon=...)` returns `+inf` for survival beyond the requested
horizon, not a claim of never defaulting. The migration generator has zero row
sums and an absorbing last/default state. Merton models terminal insolvency;
it is not a first-passage intensity model. Recovery of par in `risky_cashflows`
is paid continuously at default through the final payment time. Integration
breakpoints can be supplied for piecewise curves. [Example](examples/credit.py).

## Verification and execution

```bash
python -m pip install -e '.[dev,autodiff]'
python -m pytest -q tests/math/finance
python docs/math/examples/pricing.py
python docs/math/examples/fixed_income.py
```

The financial suite covers P/Q likelihood identities, settlement conservation,
put–call and barrier parity, analytic/tree/FD/MC comparisons, Heston's BSM limit,
synthetic volatility and native Heston calibration recovery, curve and CDS
repricing, local volatility and risk-neutral density, exact Snell recursion,
independent LSMC evaluation, hedge feasibility, bond price/yield and sensitivity
identities, Gaussian/HJM/market models, inflation bonds/swaps, structural credit,
and global-settings restoration. Tests use generated inputs, not market data.

Backend references: [QuantLib documentation](https://www.quantlib.org/docs.shtml),
[the maintainer's QuantLib guide](https://www.quantlibguide.com/), and
[PyTorch autograd](https://docs.pytorch.org/docs/2.14/autograd.html).
