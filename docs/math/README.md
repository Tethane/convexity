# Convexity math

Import modules directly from `convexity.math`. Models return Convexity result
objects; numerical work is delegated to NumPy, SciPy, scikit-learn, statsmodels,
QuantLib, CVXPY, and optional backends.

```python
from convexity.math import learning, dependence

model = learning.LinearRegression().fit([[0.0], [1.0], [2.0]], [1.0, 3.0, 5.0])
print(model.predict_mean([[3.0]]))

estimate = dependence.SampleCovariance().fit([[0.01, -0.02], [-0.01, 0.03], [0.02, 0.01]])
print(estimate.covariance)
```

For applied work, start with [predictive trading](../../examples/predictive_trading.ipynb)
and [portfolio rebalancing](../../examples/portfolio_rebalancing.ipynb). Smaller
scripts in [examples](examples) isolate individual APIs:

| Work | Modules | Example |
| --- | --- | --- |
| Regression and neural prediction | `learning`, `transforms` | [Predictive learning](examples/predictive_learning.py) |
| Time-series and volatility models | `forecasting`, `volatility` | [Forecasting](examples/forecasting.py), [volatility](examples/volatility.py) |
| Covariance and probabilistic models | `dependence`, `distributions`, `gaussian_processes` | [Dependence](examples/dependence.py), [GPs](examples/gaussian_processes.py) |
| Estimation and latent states | `estimation`, `bayesian`, `state_space`, `regimes` | [Estimation](examples/statistical_estimation.py), [state space](examples/state_space.py) |
| Portfolio decisions | `optimization`, `robust`, `risk`, `utility`, `wealth`, `diagnostics` | [Decision workflow](examples/decision_workflows.py) |
| Derivative and fixed-income models | `contracts`, `curves`, `pricing`, `implied_volatility`, `calibration`, `stopping`, `sensitivities`, `fixed_income`, `rates`, `credit`, `measures` | [Financial workflow](examples/financial_workflows.py) |
| Processes and numerical methods | `numerical`, `processes`, `rough`, `simulation` | [Numerical methods](examples/numerical_mathematics.py), [simulation](examples/simulation.py) |
| Control and execution | `control`, `policies`, `execution`, `reinforcement` | [Control](examples/optimal_control.py), [execution](examples/execution.py) |

Use `help(module.Class)` for signatures and field descriptions. Public contracts
live in each module's `interfaces`; implementations are in `builtins`. Custom
models can satisfy the public protocols without inheriting from Convexity classes.
[Shared conventions](public_contracts.md) cover shapes, units, and uncertainty.

## Optional dependencies

| Extra | Used for |
| --- | --- |
| `neural`, `autodiff` | PyTorch models and tensor differentiation |
| `forecasting` | sktime adapters |
| `forecasting-advanced` | NeuralForecast and pretrained checkpoints; may download weights |
| `state-space` | FilterPy and the pinned particles backend |
| `bayesian` | PyMC, ArviZ, and particles |
| `processes` | sdeint integration |
| `reinforcement` | PyTorch, Gymnasium, and Stable-Baselines3 |
| `notebooks` | Jupyter execution and plots |

Install the relevant extra before running its script. Backend availability does
not imply support for every native model/engine combination. Unsupported
operations raise explicitly; inspect result capabilities and convergence status.

Numerical identities and synthetic recovery tests verify implementation behavior.
They do not establish predictive accuracy, calibrated uncertainty, or trading
profitability. Use chronological validation, retain the training cutoff and data
vintage, and compare against simple baselines on untouched observations.
