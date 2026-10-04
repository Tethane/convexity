# API readiness before families 27–36

This second review tests usability and composition, not just whether each
financial adapter can call its provider. Ordinary supported workflows now use
Convexity functions, Python dates/strings, numbers, arrays and callbacks. They do
not require QuantLib imports, handles, helper constructors, engine classes,
settings changes or global fixing-store manipulation.

The executable [financial workflows example](../examples/financial_workflows.py)
covers quotes → curves → bonds/rate options → sensitivities, and rate models →
processes → scenarios → forecasts. The eleven financial notebooks use the same
public API. Native objects are retained as optional extension points; these are
not a promise that every vendor instrument is exposed through a dedicated
Convexity constructor.

## A small public workflow

```python
from convexity.math import contracts, curves, rates, fixed_income

curve = curves.bootstrap("2026-01-01", [
    curves.deposit_quote(0.03, "6M"),
    curves.deposit_quote(0.032, "1Y"),
    curves.swap_quote(0.035, "2Y"),
    curves.swap_quote(0.04, "5Y"),
])
print(curve.calibration_errors)

dates = contracts.schedule("2027-01-01", "2030-01-01", calendar="weekends")
index = rates.rate_index(curve, currency="USD", calendar="weekends")
bond = fixed_income.floating_bond(dates, index)
print(fixed_income.bond_analytics(bond, curve, compute_yield=False))

cap = rates.cap_floor(dates, index, 0.04)
print(rates.rate_option_price(cap, curve, volatility=0.20))
```

The rate-option pricer infers cap/floor versus swaption from the contract.
Users still choose economic inputs: currency, calendars, rate conventions,
volatility units, recovery, model measure and contract dates. Convexity must not
invent these financial assumptions. Standard tenors such as `"6M"` and `"5Y"`
are quote maturities, not an expression language.

## Public API additions

| Need | Public entry point |
| --- | --- |
| Deposit, fixed/IBOR swap and overnight-indexed swap calibration | `curves.deposit_quote`, `swap_quote`, `ois_quote`; feed their outputs to `bootstrap` |
| Inspect bootstrap fit without invoking vendor helpers | `curve.calibration_errors`, an owned, read-only snapshot |
| CDS calibration | `credit.cds_quote`; feed quotes to `bootstrap_cds` |
| Floating coupons with their forecast curve and history | `rates.rate_index(curve, fixings={date: rate})` |
| CPI contracts with historical observations | `rates.inflation_index(curve, fixings={date: level})` |
| Caps/floors, fixed/floating swaps and swaptions | `rates.cap_floor`, `interest_rate_swap`, `swaption`, `swap_price` |
| Inflation swaps | `rates.inflation_swap`, `inflation_swap_price` |
| Callable/putable bond exercise engine | `fixed_income.callable_bond_engine`; also usable with `functools.partial` as a key-rate-risk engine factory |
| Forward-rate simulations | `rates.forward_market_model`, then `market_model_paths` |
| Gaussian HJM bond values | `rates.gaussian_hjm`, then `gaussian_bond_price` |
| Dates suitable for ordinary Python reporting | `contracts.to_date`, `schedule_dates` |

Schedules accept calendar names (`none`, `weekends`, `target`, `us_settlement`,
`us_government_bond`, `nyse`, `uk`, `canada`) and adjustment names (`unadjusted`,
`following`, `modified_following`, `preceding`, `modified_preceding`). Bond yields
accept `simple`, `continuous` or `periodic` compounding and named frequencies
(`annual`, `semiannual`, `quarterly`, `monthly`) or supported payment counts.
Native convention objects remain accepted for existing advanced callers.

These defaults are explicit simplifications. For example, deposit/swap quote
factories default to zero settlement days and an unadjusted null calendar;
`rate_index` defaults to a six-month USD Actual/360 index with zero fixing days.
Select actual market conventions when using real quotes. `cds_quote` uses a
forward-generated premium schedule and valuation-date trade semantics, not a
standard IMM/CDS2015 contract. When repricing it with `cds_price`, supply the same
schedule and `trade_date=reference_date`. Historical trades require their actual
trade date.

## Defects and integration gaps corrected

- **Process substitutability:** `ShortRateModel.process()` formerly returned a
  raw QuantLib process for Hull–White, although Vasicek/CIR returned Convexity
  processes. Hull–White now provides the existing transition/mean/covariance/
  sampling contract. G2 returns the existing two-dimensional Gaussian diffusion
  over its factors. The curve-dependent deterministic short-rate shift is not
  silently added to those factor states.
- **NumPy input interoperability:** validated NumPy integer engine settings still
  failed inside SWIG. NumPy float32 spots, strikes and rate-model parameters also
  failed native constructors. Public objects now normalize these validated
  numerical inputs to native Python scalars.
- **Currency integrity:** pricing dispatch rejected neither a USD result for a
  EUR contract nor missing currency propagation from rate indices. Custom
  engine results are checked against contract currency; public rate instruments
  propagate their index currency into results.
- **Fixing-history ownership:** supplied mappings are copied and applied only
  during Convexity valuation. Existing native histories are restored on success,
  exceptions and nested calls, under the shared lock. Tests alternate two
  histories for the same index and verify unchanged external histories.
- **Fixing lag:** floating bonds now inherit the index's fixing days unless
  explicitly overridden. An index with two-day fixing lag no longer silently
  produces zero-day-fixing coupons. Pass `fixing_days=0` explicitly to retain
  that choice.
- **Engine capability checks:** wrong rate-option kinds, non-European exercise
  with European-only engines, and ignored normal-model displacement fail at the
  Convexity boundary rather than being silently accepted or producing opaque
  vendor errors.
- **Boundary consistency:** negative curve date-times fail like negative elapsed
  times; empty dividend generators work as empty cash-flow collections;
  nonsquare migration generators and negative default-state exposures fail
  cleanly before matrix arithmetic.

There was also a useful convention counterexample: a bootstrapped CDS repriced
with an inferred historical trade date differed by about `3.89e-5` per unit
notional, exactly its one-day premium rebate. With matching explicit trade dates
it reprices to numerical zero. The test and example now specify that economic
input rather than treating the difference as a solver error.

## Composition contracts for the next families

| Planned families | Existing inputs to reuse |
| --- | --- |
| 27–28: risk and utility | `Distribution` expectations/quantiles, `PathScenarios` whole-path weights and payoff mappings, explicit P/Q models |
| 29–30: optimization | Ordinary NumPy exposure matrices, callback objectives, calibrated parameters and bounded hedge results; no vendor object required for a supplied numerical problem |
| 31–32: control and policies | `StochasticProcess.transition`, exact/sample-only kernel capabilities, finite-state stopping problems and independently evaluated stopping policies |
| 33–34: execution and wealth | Payoff/delivery conventions, explicit cash flows and currencies, prices and per-unit sensitivities |
| 35–36: diagnostics and optional RL | Joint forecast distributions, scenario sampling provenance, owned array results, existing fit/predict and transition interfaces |

New integration tests verify that scenario-to-forecast conversion preserves
joint covariance, sparse selection preserves the selected covariance block,
payoff-distribution expectations match discounted scenario prices, importance
reweighting removes unsupported IID error claims, and option Greeks feed hedge
geometry directly. Every public export in all 26 families resolves.

Future APIs should follow this boundary: ordinary arrays/functions for supplied
mathematical problems, optional model/result objects when they preserve actual
meaning, and provider-specific syntax confined to explicit extensions. There is
no need for a parallel modeling language or mandatory registry.

## Verification

The final verification results are:

| Check | Result |
| --- | --- |
| Full repository suite in configured `quant` | **1,556 passed, no failures or skips**, 250.43 seconds |
| Focused financial/API/composition suite | **157 passed** on both Python 3.14.7 and Python 3.12.3 |
| Whole-repository Pyright | **0 errors, 0 warnings** in `quant` |
| Notebook-builder Pyright | **0 errors, 0 warnings** |
| Lint on changed financial code, tests, examples and notebook builder | E4/E7/E9/F checks pass; existing wildcard-import and lambda test conventions excluded |
| Fresh-kernel notebook execution | **25 of 25 passed**; all eleven financial notebooks retain updated output |
| Executable financial examples | **12 of 12 passed**, including the new combined workflow |
| Primary environment dependency consistency | `python -m pip check`: **no broken requirements** |
| Whitespace/link/output checks | Clean diff; local guide links resolve; financial notebooks contain no native imports/method calls or error outputs |

The primary environment has Python 3.14.7, QuantLib 1.43, NumPy 2.4.6, SciPy
1.18.1, pandas 3.0.5 and PyTorch 2.14.0. Optional integrations exercised there
include NeuralForecast 3.2.2, sktime 1.2.0, FilterPy 1.4.5, particles 0.4 and
PyMC 5.28.5. The secondary Python 3.12 environment supplies an additional check
of the changed financial and integration boundaries.

The full suite reports 70 warnings, not a warning-free execution. They include
neural-training configuration/performance notices, an sktime future-default
notice, and intentionally singular/overflow/ill-conditioned numerical test
cases. NeuralForecast also warns when converting a pandas read-only NumPy view:
its inspected `_as_torch_copy` implementation immediately clones the tensor
before using it. No global warning suppression or vendor monkeypatch was added.
These run results establish tested behavior, not a guarantee for arbitrary
future dependency releases or live-market inputs.

Financial and composition tests are in
`tests/math/finance/` and `tests/math/test_readiness_composition.py`. The workflow
example runs as a regular Python script. The full run uses the IDE-configured
Conda `quant` interpreter, including optional forecasting/state-space packages.
Reproduce the main checks from the repository root with the configured interpreter:

```bash
conda run -n quant python -m pytest -q
conda run -n quant python -m pyright
conda run -n quant python -m pip check
conda run -n quant python examples/financial_workflows.py
conda run -n quant python scripts/build_financial_notebooks.py --execute
```

Notebook validation uses fresh kernels; financial notebooks retain updated
outputs, while the other notebooks are executed for QA without replacing their
existing saved outputs.

## Readiness limits

The existing [financial catalog](financial_mathematics.md) remains the capability
boundary. Callable-bond carry is still explicitly unsupported; surface arbitrage
checks are sampled; LSMC provides a feasible-policy estimate rather than a dual
bound. P forecasts are never automatically converted into Q laws. Native object
mutation and calls outside Convexity's valuation contexts remain advanced caller
responsibilities. These explicit limits do not require a new API foundation for
families 27–36, but new features must not imply those missing capabilities exist.

Provider rationale: QuantLib's maintainer explains its
[handle/dependency design](https://www.quantlibguide.com/Handling%20dependencies.html)
and [calendar conventions](https://www.quantlibguide.com/Calendars%20and%20holidays.html).
Convexity keeps those mechanisms in its backend while exposing the financial
choices needed by the caller.
