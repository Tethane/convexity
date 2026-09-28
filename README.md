# Convexity

A generic mathematical framework for future trading systems. The current scope
is the seven completed foundation families in [the roadmap](docs/direction.md).
Time-series forecasting (family 8) is next.

Each family provides mathematical contracts, working library adapters, user
extension points, numerical tests and examples with supplied or generated data.
NumPy, SciPy, scikit-learn, statsmodels and findiff supply the numerical work;
PyTorch is optional.

| Family | Module | Catalog and contracts | Example |
|---|---|---|---|
| 1. Numerical mathematics | `convexity.math.numerical` | [Guide](docs/numerical_mathematics.md) | [Script](examples/numerical_mathematics.py) |
| 2. Probability distributions | `convexity.math.distributions` | [Guide](docs/distributions.md) | [Script](examples/distributions.py) |
| 3. Statistical estimation | `convexity.math.estimation` | [Guide](docs/statistical_estimation.md) | [Script](examples/statistical_estimation.py) |
| 4. Transforms and factor geometry | `convexity.math.transforms` | [Guide](docs/transforms.md) | [Script](examples/transforms.py) |
| 5. Covariance, dependence and tails | `convexity.math.dependence` | [Guide](docs/dependence.md) | [Script](examples/dependence.py) |
| 6. Regression and predictive learning | `convexity.math.learning` | [Guide](docs/predictive_learning.md) | [Script](examples/predictive_learning.py) |
| 7. Gaussian processes and kernels | `convexity.math.gaussian_processes` | [Guide](docs/gaussian_processes.md) | [Script](examples/gaussian_processes.py) |

Install in a Python 3.10+ environment and run an example:

```bash
python -m pip install -e '.[dev]'
python examples/gaussian_processes.py
```

```python
from convexity.math import distributions as dist, numerical as num

solution = num.solve([[3., 1.], [1., 2.]], [9., 8.])
print(solution.value)
print(dist.normal(loc=2., scale=3.).quantile([.05, .5, .95]))
```

Install `.[autodiff]` for tensor laws and automatic differentiation, `.[neural]`
for neural predictors, or `.[notebooks]` for the seven worked
[notebooks](notebooks/). Notebook outputs are generated locally and kept out of
source control.

Run the full suite and type check, or scope tests to the family being changed:

```bash
python -m pytest -q
python -m pyright --pythonpath "$(command -v python)"
python -m pytest -q tests/math/gaussian_processes
```

`convexity/math/` contains the library, organized into public `interfaces/` and
`builtins/` with private `_backends/`. Tests live in `tests/math/`; executable
examples, notebooks and documentation live outside the package. Add future
families only with a bounded catalog and the acceptance checks in the roadmap.

The [mathematical system specifications](docs/mathematical_trading_systems/)
provide reference requirements for later families. Trading runtimes, broker
integrations and market-data acquisition are outside the current repository scope.
