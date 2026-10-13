# Convexity math

This is the mathematical library guide. Commands below run from the repository
root.

A generic mathematical framework for future trading systems. The current scope
covers families 1–36 in [the roadmap](direction.md), including time-series
and volatility forecasting, state-space models, Bayesian parameter inference,
regimes, event models, stochastic differential equations, rough processes,
simulation, joint path distributions, QuantLib-based financial mathematics,
risk, utility, optimization, control, execution, wealth accounting and diagnostics.
Reinforcement learning remains optional.

Each family provides mathematical contracts, working library adapters, user
extension points, numerical tests and examples with supplied or generated data.
NumPy, SciPy, scikit-learn, statsmodels, arch, findiff, QuantLib, CVXPY and
python-control supply the numerical work;
PyTorch is optional.

Start with the [public API guide and readiness review](api_readiness.md) and
[financial workflow script](examples/financial_workflows.py). Supported financial
workflows require no QuantLib imports, handles, helper constructors or global settings.

| Family | Module | Catalog and contracts | Example |
|---|---|---|---|
| 1. Numerical mathematics | `convexity.math.numerical` | [Guide](numerical_mathematics.md) | [Script](examples/numerical_mathematics.py) |
| 2. Probability distributions | `convexity.math.distributions` | [Guide](distributions.md) | [Script](examples/distributions.py) |
| 3. Statistical estimation | `convexity.math.estimation` | [Guide](statistical_estimation.md) | [Script](examples/statistical_estimation.py) |
| 4. Transforms and factor geometry | `convexity.math.transforms` | [Guide](transforms.md) | [Script](examples/transforms.py) |
| 5. Covariance, dependence and tails | `convexity.math.dependence` | [Guide](dependence.md) | [Script](examples/dependence.py) |
| 6. Regression and predictive learning | `convexity.math.learning` | [Guide](predictive_learning.md) | [Script](examples/predictive_learning.py) |
| 7. Gaussian processes and kernels | `convexity.math.gaussian_processes` | [Guide](gaussian_processes.md) | [Script](examples/gaussian_processes.py) |
| 8. Time-series forecasting | `convexity.math.forecasting` | [Guide](time_series_forecasting.md) | [Script](examples/forecasting.py) |
| 9. Volatility forecasting | `convexity.math.volatility` | [Guide](volatility_forecasting.md) | [Return models](examples/volatility.py), [HAR](examples/volatility_har.py) |
| 10. State-space inference | `convexity.math.state_space` | [Guide](state_space.md) | [Linear](examples/state_space.py), [Nonlinear/particles](examples/state_space_nonlinear.py), [Estimation](examples/state_space_estimation.py) |
| 11. Bayesian inference | `convexity.math.bayesian` | [Guide](bayesian_inference.md) | [Conjugate](examples/bayesian.py), [PyMC](examples/bayesian_pymc.py), [Joint sequential](examples/bayesian_sequential.py) |
| 12. Regimes, catalysts and events | `convexity.math.regimes` | [Guide](regimes_events.md) | [Regimes](examples/regimes.py), [Events](examples/events.py) |
| 13. Stochastic processes and differential equations | `convexity.math.processes` | [Guide](stochastic_processes.md) | [Exact models](examples/processes.py), [Numerical backends](examples/processes_backends.py) |
| 14. History-dependent and rough processes | `convexity.math.rough` | [Guide](rough_processes.md) | [Script](examples/rough.py) |
| 15. Simulation and joint path distributions | `convexity.math.simulation` | [Guide](simulation.md) | [Script](examples/simulation.py) |
| 16. Physical and pricing measures | `convexity.math.measures` | [Guide](financial_mathematics.md) | [Script](examples/measures.py) · [Notebook](notebooks/measures.ipynb) |
| 17. Payoffs, cash flows and contracts | `convexity.math.contracts` | [Guide](financial_mathematics.md) | [Script](examples/contracts.py) · [Notebook](notebooks/contracts.ipynb) |
| 18. Curves, forwards, basis and carry | `convexity.math.curves` | [Guide](financial_mathematics.md) | [Script](examples/curves.py) · [Notebook](notebooks/curves.ipynb) |
| 19. Derivative pricing engines | `convexity.math.pricing` | [Guide](financial_mathematics.md) | [Script](examples/pricing.py) · [Notebook](notebooks/pricing.ipynb) |
| 20. Implied volatility and surfaces | `convexity.math.implied_volatility` | [Guide](financial_mathematics.md) | [Script](examples/implied_volatility.py) · [Notebook](notebooks/implied_volatility.ipynb) |
| 21. Calibration and inverse problems | `convexity.math.calibration` | [Guide](financial_mathematics.md) | [Script](examples/calibration.py) · [Notebook](notebooks/calibration.ipynb) |
| 22. Optimal stopping and early exercise | `convexity.math.stopping` | [Guide](financial_mathematics.md) | [Script](examples/stopping.py) · [Notebook](notebooks/stopping.ipynb) |
| 23. Sensitivities and hedging | `convexity.math.sensitivities` | [Guide](financial_mathematics.md) | [Script](examples/sensitivities.py) · [Notebook](notebooks/sensitivities.ipynb) |
| 24. Bonds and fixed income | `convexity.math.fixed_income` | [Guide](financial_mathematics.md) | [Script](examples/fixed_income.py) · [Notebook](notebooks/fixed_income.ipynb) |
| 25. Rates and inflation | `convexity.math.rates` | [Guide](financial_mathematics.md) | [Script](examples/rates.py) · [Notebook](notebooks/rates.ipynb) |
| 26. Credit and default | `convexity.math.credit` | [Guide](financial_mathematics.md) | [Script](examples/credit.py) · [Notebook](notebooks/credit.ipynb) |
| 27. Risk measures and stress | `convexity.math.risk` | [Guide](decision_mathematics.md) | [Script](examples/risk.py) · [Notebook](notebooks/risk.ipynb) |
| 28. Utility and decisions | `convexity.math.utility` | [Guide](decision_mathematics.md) | [Script](examples/utility.py) · [Notebook](notebooks/utility.ipynb) |
| 29. Mathematical optimization | `convexity.math.optimization` | [Guide](decision_mathematics.md) | [Script](examples/optimization.py) · [Notebook](notebooks/optimization.ipynb) |
| 30. Robust and stochastic optimization | `convexity.math.robust` | [Guide](decision_mathematics.md) | [Script](examples/robust.py) · [Notebook](notebooks/robust.ipynb) |
| 31. Dynamic programming and control | `convexity.math.control` | [Guide](decision_mathematics.md) | [Script](examples/optimal_control.py) · [Notebook](notebooks/control.ipynb) |
| 32. Threshold and hysteresis policies | `convexity.math.policies` | [Guide](decision_mathematics.md) | [Script](examples/policies.py) · [Notebook](notebooks/policies.ipynb) |
| 33. Execution and assignment | `convexity.math.execution` | [Guide](decision_mathematics.md) | [Script](examples/execution.py) · [Notebook](notebooks/execution.ipynb) |
| 34. Financed gains and wealth | `convexity.math.wealth` | [Guide](decision_mathematics.md) | [Script](examples/wealth.py) · [Notebook](notebooks/wealth.ipynb) |
| 35. Diagnostics and model comparison | `convexity.math.diagnostics` | [Guide](decision_mathematics.md) | [Script](examples/diagnostics.py) · [Notebook](notebooks/diagnostics.ipynb) |
| 36. Reinforcement learning (optional) | `convexity.math.reinforcement` | [Guide](decision_mathematics.md) | [Script](examples/reinforcement.py) · [Notebook](notebooks/reinforcement.ipynb) |

Try the [combined decision workflow](examples/decision_workflows.py) for physical
scenarios, a Q option premium, financed wealth, ES constraints and utility selection.

Install in the Python 3.14 `quant` environment and run an example:

```bash
python -m pip install -e '.[dev]'
python docs/math/examples/forecasting.py
```

```python
from convexity.math import distributions as dist, numerical as num

solution = num.solve([[3., 1.], [1., 2.]], [9., 8.])
print(solution.value)
print(dist.normal(loc=2., scale=3.).quantile([.05, .5, .95]))
```

Install `.[autodiff]` for tensor laws and automatic differentiation, `.[neural]`
for neural predictors, sequence forecasters and generative models; `.[forecasting-advanced]`
for dedicated architectures and pretrained checkpoints; `.[forecasting]` for
optional sktime interoperability; `.[state-space]` for FilterPy and particles
inference (see the [backend pin](state_space.md#installation-and-execution));
`.[bayesian]` for PyMC, ArviZ and joint parameter/state particle inference;
`.[processes]` for sdeint custom SDE integration (QuantLib is a core dependency);
`.[reinforcement]` for optional PyTorch/Stable-Baselines3 actor–critic methods;
or `.[notebooks]` for notebook execution and plotting. Financial audit findings
and supported-engine limits are recorded in [the QA review](financial_qa.md).
The thirty-five worked [notebooks](notebooks/README.md) retain executed plots and tables for immediate
inspection; the notebook guide lists the additional extras used by their examples.
The [forecasting completion record](forecasting_completion.md)
lists the expanded catalog and real-checkpoint verification.
The [mathematical audit](mathematical_audit.md) distinguishes analytic
verification, backend execution and predictive-quality evidence, and records
the defects found by additional adversarial tests.

The [public contract conventions](public_contracts.md) and
[generated reference for all 36 families](reference/README.md) document
input/output types, axes, uncertainty semantics and extension points. Verify
reference freshness with `python scripts/build_math_reference.py --check`.

The [36-family adversarial QA report](sdk_adversarial_audit.md) records
cross-family checks, corrected numerical defects, measured coverage and the
remaining production-readiness blockers. Reproduce the public API inventory
with `python scripts/audit_math_sdk.py --output /tmp/convexity-api.json`.

The local development interpreter is Conda's `quant` environment, also selected
in `.vscode/settings.json`. Install the full set of test dependencies there:

```bash
conda run -n quant python -m pip install -e '.[dev,autodiff,neural,forecasting,forecasting-advanced,state-space,bayesian,processes,reinforcement,notebooks]'
conda run -n quant python -m pip check
conda run -n quant python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
conda run --no-capture-output -n quant python -m pytest -q -rs
conda run --no-capture-output -n quant python -m pyright
```

The CUDA check verifies PyTorch's runtime in that interpreter. The CUDA compiler
reported by `nvcc` is a separate installation. Review pytest's skip summary when
validating the full catalog; missing optional integrations are not verified tests.

Run the full suite and type check, or scope tests to the family being changed:

```bash
python -m pytest -q
python -m pyright --pythonpath "$(command -v python)"
python -m pytest -q tests/math/forecasting
python -m pytest -q tests/math/volatility
python -m pytest -q tests/math/state_space
python -m pytest -q tests/math/bayesian
python -m pytest -q tests/math/regimes
python -m pytest -q tests/math/processes
python -m pytest -q tests/math/rough
python -m pytest -q tests/math/simulation
python -m pytest -q tests/math/finance
python -m pytest -q tests/math/decisions
```

`convexity/math/` contains the library, organized into public `interfaces/` and
`builtins/` with private `_backends/`. Tests live in `tests/math/`; executable
examples, notebooks and documentation live outside the package. Add future
families only with a bounded catalog and the acceptance checks in the roadmap.

The [mathematical system specifications](mathematical_trading_systems)
provide reference requirements for the implemented catalogs. Trading runtimes, broker
integrations and market-data acquisition are outside the mathematical library's scope.
