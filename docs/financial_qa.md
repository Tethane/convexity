# Financial mathematics engineering and QA review

The later [API readiness review](api_readiness.md) extends this first pass with
public workflows, integration checks and validation in the configured Conda environment.

This review covers families 16–26 and their integration with the existing core.
The original 59 financial tests passed, but the first adversarial pass reproduced
**13 failures in 14 tests**. Passing examples and vendor delegation were not
sufficient evidence of correct financial semantics or consistent architecture.
The corrected financial suite has **107 passing tests**, including 29 adversarial
regressions and 19 architecture checks. All inputs are generated locally.

## Architecture corrections

All eleven families now have three functional layers:

- `interfaces/` owns problems, protocols and validated result records.
- `builtins/` owns public model/configuration objects and composition with the
  existing mathematical core.
- `_backends/` owns QuantLib, SciPy and scikit-learn execution. Backend functions
  are private implementation details; public classes remain in public layers.

The shared QuantLib date conversion and settings context lives in
`convexity/math/_backends/quantlib.py`. `_finance.py` provides shared validation
and the convention bridge. QuantLib enums and native constructors no longer
leak through direct vendor imports in interfaces or builtins. Native objects
remain explicit extension points, with convention defaults resolved by providers.
Small payoff formulas and validated model equations remain in builtins; backend
folders are not empty placeholders.

Result records now include `CDSResult`, `HazardEstimate`,
`StructuralCreditResult`, `PolicyEvaluation`, `NativeCalibrationResult`,
`RatePaths` and `CarryResult`. Numerical arrays are copied and made read-only;
keyed access remains available for callers using the previous dictionary fields.
Native vendor objects and fitted regressions retain their vendor mutability.
Architecture tests prohibit direct specialized-vendor imports in public layers
and check public ownership of representative models and results. A custom
pricing engine still works without vendor inheritance or registration.

## Material findings and corrections

| Finding | Correction and evidence |
| --- | --- |
| Callable-bond analytics and curve risk could value promised cash flows while ignoring exercise. | Require an explicit exercise engine. Key-rate duration accepts an `engine_factory(curve)` and reprices the option for every bump; summed key-rate risk is checked against parallel effective duration. Callable carry fails explicitly until an exercise-aware carry policy is supplied by a future implementation. |
| Binomial vanilla pricing accepted shaped curves but QuantLib internally replaced them with flat expiry rates. | Require proven constant coefficients for the tree. A generated American put priced at approximately 6.40246 with the old tree versus 6.11295 with the shaped-curve finite-difference engine. Analytic barrier and geometric-Asian engines receive the same constant-coefficient guard. |
| Surface-based finite differences used a strike-frozen Black-volatility PDE by default. | Request QuantLib local volatility explicitly. A constant-volatility surface recovers the analytic benchmark. Inconsistent surfaces still fail; local variance is not silently repaired. |
| Basket assets could use distinct discount curves that happened to agree only at expiry. | Require a shared native discount curve or matching proven flat curves. One-step basket and fixing-grid arithmetic-Asian MC require constant volatility; a smile needs a time-resolved path engine. |
| Zero-volatility analytic results reported misleading zero Greeks at a payoff kink. | Preserve the price and omit Greeks for scalar/native constant zero volatility. Engines requiring positive volatility reject the degenerate input. |
| Calendar diagnostics sampled a uniform time grid and could miss a narrow arbitrage interval. | Always include every original expiry knot and intermediate points. A 1.000/1.001/1.002-year counterexample is detected. Empty reports and fewer than two time samples are rejected. |
| Risk-neutral density rejected negative strikes, excluding normal-rate models. | Permit real strike grids; finite differences recover an independently specified normal density. |
| Terminal numeraires, transition probabilities and stopping-policy inputs were insufficiently validated. | Require positive finite numeraires, tight stochastic-row tolerance, nonnegative LSMC evaluation rewards, and aligned policy dimensions/exercise dates. |
| QuantLib seed zero selected an automatic seed, and American MC training did not receive its own fixed seed. | Require positive seeds; explicitly seed both calibration and valuation streams. Repeated American MC calls reproduce their price. |
| Asian running accumulators could contradict the declared past-fixing count. | Require neutral accumulators with no past observations and positive sums/products when past fixings exist. |
| Pricing processes mixed incompatible year-fraction clocks; rate options could combine a fitted model and a different curve. | Validate discount/dividend/volatility clocks, Heston curve clocks, fitted rate-model curve identity, and nonnegative CIR states. |
| Custom calibrators could return valid-looking results for the wrong problem. | Check observation/parameter axes and bounds at dispatch. Covariance uses the Jacobian SVD rather than normal equations, preserving numerical conditioning. |
| Standalone floating-bond price/yield calculations depended on ambient QuantLib settings. | Add explicit `valuation_date`, defaulting to settlement; scope calculation and restore global state. Test with conflicting surrounding dates and a future-start floating bond. |
| CDS trade-date semantics were implicit and result dictionaries did not enforce leg reconciliation. | Add trade/protection/settlement parameters, include accrued-premium rebate in the net premium leg, expose it separately, and compute spread risk by affine repricing. Current-trade and zero-coupon-spread tests reprice fair spread to zero. QuantLib schedule/protection conventions still apply. |

The CDS rebate case was an extension of the audit, not one of the initial 13
failures: the original default-trade-date example had an already-settled rebate.
The new explicit current-trade test exercises a nonzero rebate.

The coefficient restrictions follow the actual QuantLib algorithms:
[binomial engine](https://raw.githubusercontent.com/lballabio/QuantLib/master/ql/pricingengines/vanilla/binomialengine.hpp),
[analytic barrier engine](https://raw.githubusercontent.com/lballabio/QuantLib/master/ql/pricingengines/barrier/analyticbarrierengine.cpp),
and [geometric Asian engine](https://raw.githubusercontent.com/lballabio/QuantLib/master/ql/pricingengines/asian/analytic_discr_geom_av_price.cpp).
Global settings and fixing behavior are described in the maintainer's
[evaluation-date guide](https://www.quantlibguide.com/The%20global%20evaluation%20date.html).

## Notebook evidence

Eleven executed notebooks, one per financial family, retain explanatory prose,
plots, tables and generated-data comparisons. The [notebook index](../notebooks/README.md)
describes their experiments. They include P/Q reweighting, schedule accruals,
bootstrap quote repricing, engine convergence, surface arbitrage, inverse-problem
identifiability, independently evaluated exercise policies, hedge nullspaces,
callable curve risk, rate measures and credit leg reconciliation.

Rebuild and execute with the intended Python environment:

```bash
python scripts/build_financial_notebooks.py --execute
python scripts/build_financial_notebooks.py --execute --only pricing credit
python -m pytest -q tests/math/finance
```

The builder overwrites the selected financial notebooks and runs fresh kernels
with the invoking interpreter. It does not download datasets or checkpoints.
Notebooks contain experiments rather than duplicated assertion suites.

## Verification environment

Validation used Python 3.12.3, QuantLib 1.43, NumPy 2.5.3 and SciPy 1.18.1.
The final full repository run completed with **1,488 passed, 13 skipped and
3 warnings** in 211.91 seconds. The warnings arise in existing learning/numerical
tests; no financial tests failed or skipped. The financial suite contributes
107 passing cases.
The focused Pyright run covers all eleven families, shared finance/QuantLib
helpers, financial tests, eleven example scripts and the notebook builder:
**zero errors and zero warnings**. Ruff's E4/E7/E9/F checks pass on financial
code, tests and builder (existing wildcard-import test conventions excluded).
All eleven standalone financial scripts execute successfully.

All eleven notebooks were executed in fresh kernels after the final changes:
**47 code cells, 34 retained plot outputs and 28 retained table outputs**.
Notebook schema, execution order, absence of error outputs and presence of plots
and tables were checked; representative convergence, surface, accrual and
exercise-region plots were visually inspected.

The whole-repository type-check attempt also encounters missing optional
forecasting/state-space packages in this environment (`neuralforecast`,
`huggingface_hub`, `chronos`, `timesfm`, `filterpy`, `sktime`). The clean focused
result does not claim those unrelated optional integrations were type-checked.

## Remaining boundaries

This review does not certify arbitrary native QuantLib instrument/engine pairings
or every possible model and convention. The [financial catalog](financial_mathematics.md)
states the implemented combinations. In particular:

- Arbitrage diagnostics are sampled checks, not global certificates.
- LSMC reports a feasible-policy estimate and conditional IID error; it does not
  produce a dual upper bound. Independent validation paths remain essential.
- Native GSR/market-model simulations retain their terminal-forward measure;
  empirical curves and statistical default estimates retain P labels.
- Callable carry is explicitly unsupported; promised-cash-flow yield duration
  is not effective option-adjusted duration.
- Native fixing stores and QuantLib objects are mutable. The shared lock protects
  cooperating Convexity calls, not arbitrary external threads changing settings.
- Greeks absent from an engine remain absent; PyTorch differentiates tensor-native
  callbacks, not QuantLib's native calls.

Convergence tests and generated financial identities establish behavior on the
specified cases. They do not establish calibration quality on live market data.
