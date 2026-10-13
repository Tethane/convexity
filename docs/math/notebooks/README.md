# Inspecting Convexity

These notebooks use Convexity's public interfaces to investigate mathematical
behavior. They retain executed plots and tables so the results can be inspected
without first running a kernel. Generated data, specified laws, and reference
solutions make assumptions and errors visible. Assertions, adapter scaffolding,
and artifact round-trip checks belong in the Python test suite.

| Notebook | What to inspect |
| --- | --- |
| [Numerical mathematics](numerical_mathematics.ipynb) | Conditioning versus residuals, iteration budgets, derivatives, interpolation nodes, stiffness, and mesh refinement. |
| [Distributions](distributions.ipynb) | Equal-variance tails, finite scenarios, mixtures, conditioning, and change of variables. |
| [Statistical estimation](statistical_estimation.ipynb) | Parameter recovery, uncertainty coordinates, precision weighting, contamination, constraints, and bootstrap variability. |
| [Transforms](transforms.ipynb) | Training-only scaling, outlier effects, held-out reconstruction, whitening, lag alignment, and factor recovery. |
| [Dependence](dependence.ipynb) | Covariance shrinkage, changing correlation, matrix repair, copulas, and extreme-value assumptions. |
| [Predictive learning](predictive_learning.ipynb) | Held-out errors, exposure, class probabilities, conditional quantile coverage, and extrapolation. |
| [Gaussian processes](gaussian_processes.ipynb) | Length scales, data gaps, latent versus observation uncertainty, and correlated posterior draws. |
| [Forecasting](forecasting.ipynb) | Oracle AR forecasts, joint paths, rolling-origin errors, real observation updates, scenarios, and multiple targets. |
| [Volatility](volatility.ipynb) | Return versus variance uncertainty, GARCH recovery, variance scoring, and correlated lognormal HAR forecasts. |
| [State space](state_space.ipynb) | Missing measurements, irregular elapsed times, filtering versus smoothing, noise sensitivity, nonlinear approximations, and particle paths. |
| [Bayesian inference](bayesian.ipynb) | Posterior contraction, parameter versus observation uncertainty, conjugate predictive checks, and correlated regression predictions. |
| [Regimes and events](regimes.ipynb) | Hidden regimes, causal versus retrospective probabilities, dependent switching forecasts, scheduled surprises, and event clustering. |
| [Stochastic processes](processes.ipynb) | Exact transition laws, coupled OU geometry, CIR boundaries, jump tails, Heston paths, and custom Itô dynamics. |
| [History-dependent and rough processes](rough.ipynb) | Fractional versus Volterra histories, rough Bergomi paths, conditional forward variance, common-noise factors, and integrated kernel error. |
| [Physical and pricing measures](measures.ipynb) | P/Q likelihood reweighting, effective sample size, numeraires and analytic pricing. |
| [Payoffs and contracts](contracts.ipynb) | Stub accruals, day counts, settlement conservation, multipliers and corporate actions. |
| [Curves and carry](curves.ipynb) | Discount and forward shapes, bootstrap quote repricing and carrying costs. |
| [Derivative pricing](pricing.ipynb) | Parity, Greeks, tree/FD convergence, Monte Carlo error and Heston/path claims. |
| [Implied volatility and surfaces](implied_volatility.ipynb) | SVI inversion/fitting, narrow calendar arbitrage, densities and Dupire volatility. |
| [Calibration](calibration.ipynb) | Heteroskedastic quote weights, parameter recovery, identifiability, bounds and regularization. |
| [Optimal stopping](stopping.ipynb) | Snell envelopes, independent LSMC evaluation, exercise regions and dividends. |
| [Sensitivities and hedging](sensitivities.ipynb) | Bump-size error, cross-Greeks, exposure geometry and nullspaces. |
| [Bonds and fixed income](fixed_income.ipynb) | Yield geometry, accrued interest, amortization, callable effective risk and carry. |
| [Rates and inflation](rates.ipynb) | P/Q short rates, terminal-forward simulations, inflation dynamics and cap engines. |
| [Credit and default](credit.ipynb) | Survival censoring, recovery, CDS rebate/spread risk, migration and structural P/Q default. |
| [Risk measures and stress](risk.ipynb) | Family 27: generated-data workflows and numerical investigations; [contracts](../decision_mathematics.md). |
| [Utility and decisions](utility.ipynb) | Family 28: generated-data workflows and numerical investigations; [contracts](../decision_mathematics.md). |
| [Mathematical optimization](optimization.ipynb) | Family 29: generated-data workflows and numerical investigations; [contracts](../decision_mathematics.md). |
| [Robust and stochastic optimization](robust.ipynb) | Family 30: generated-data workflows and numerical investigations; [contracts](../decision_mathematics.md). |
| [Dynamic programming and control](control.ipynb) | Family 31: generated-data workflows and numerical investigations; [contracts](../decision_mathematics.md). |
| [Threshold and hysteresis policies](policies.ipynb) | Family 32: generated-data workflows and numerical investigations; [contracts](../decision_mathematics.md). |
| [Execution and assignment](execution.ipynb) | Family 33: generated-data workflows and numerical investigations; [contracts](../decision_mathematics.md). |
| [Financed gains and wealth](wealth.ipynb) | Family 34: generated-data workflows and numerical investigations; [contracts](../decision_mathematics.md). |
| [Diagnostics and model comparison](diagnostics.ipynb) | Family 35: generated-data workflows and numerical investigations; [contracts](../decision_mathematics.md). |
| [Reinforcement learning (optional)](reinforcement.ipynb) | Family 36: generated-data workflows and numerical investigations; [contracts](../decision_mathematics.md). |

## Run and retain results

Use conda `quant` and install the editable package from the repository root:

```bash
conda activate quant
python -m pip install -e '.[notebooks,autodiff,neural,state-space,processes,reinforcement]'
```

The tensor, automatic-differentiation, and neural sections require PyTorch.
The state-space notebook requires FilterPy and particles; the `state-space`
extra includes the [documented particles revision](../state_space.md#installation-and-execution).
QuantLib is a principal dependency; the process notebook additionally uses
sdeint from `.[processes]`. The decision notebooks use the principal CVXPY and
python-control dependencies; the RL notebook additionally requires `.[reinforcement]`.
All thirty-five notebooks use local generated data and require no external datasets or
model checkpoints. Choose the `quant` interpreter in your notebook editor, restart
the kernel, run all cells in order, and save the executed notebook.

To regenerate and execute the eleven financial notebooks with the invoking
interpreter (overwriting those notebooks):

```bash
python scripts/build_financial_notebooks.py --execute
```

Use `--only pricing credit` to select particular financial notebooks. See the
[financial QA record](../financial_qa.md) for defects these experiments and
regression tests exposed, supported-engine restrictions and validation results.

For a fresh-kernel batch execution of the whole catalog from the repository root:

```python
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path.cwd()
for path in sorted((root / "docs" / "math" / "notebooks").glob("*.ipynb")):
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
    ).execute()
    nbformat.write(notebook, path)
```

Run that Python code with the `quant` interpreter. Seeds make the generated
experiments reproducible; optimizer and floating-point results can still vary
across dependency versions. Outputs are examples of fitted results, not fixed
acceptance thresholds. Run `python -m pytest -q` for the automated contracts.

The eleven financial notebooks now use only public Convexity operations for
curves, indices, contracts and engines; direct QuantLib calls are unnecessary.
The [API readiness review](../api_readiness.md) records the second audit.

## Interpretation corrected during review

The estimation example now generates heteroskedastic data before applying
precision weights. Gaussian and Student-t conditional comparisons now start
from matching unconditional covariances. Neural and tree fits expose behavior
outside their training domain. Forecast updates assimilate actual new
observations and compare the same future timestamps. The forecasting notebook
also demonstrates AR's joint covariance rather than implying only marginal
uncertainty is available.

The review exposed an implementation gap in analytic log-HAR: the recursive log
covariance was available internally but only marginal moments were returned.
It now exposes a correlated lognormal law with exact marginal intervals. The
shared forecast result preserves a supplied joint law when scalar marginal laws
are supplied alongside it; sparse selection preserves their joint dependence.
Mathematical regression tests cover the covariance, change-of-variables density,
quantiles, sampling, and deterministic limit.

The notebooks distinguish conditional model uncertainty from parameter
uncertainty, pointwise intervals from path bands, and a retrospective smoother
from an online filter. Generated truth supports comparisons within each
experiment; it does not establish a universal ranking of model families.

Regenerate the ten decision notebooks with fresh kernels using:

```bash
python scripts/build_decision_notebooks.py --execute
```

Use `--only risk control` to select particular families.
