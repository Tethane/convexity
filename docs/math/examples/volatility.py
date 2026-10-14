"""Generated-return volatility catalog, innovation laws, paths, and evaluation."""

from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

from convexity.math import forecasting as ts, volatility as vol


def generated_returns(n=1800, seed=42):
    rng = np.random.default_rng(seed)
    y, variance = np.empty(n), np.empty(n)
    h, previous = 1.0, 0.0
    for t in range(n):
        h = 0.06 + 0.10 * previous**2 + 0.84 * h
        previous = np.sqrt(h) * rng.normal()
        y[t], variance[t] = previous, h
    return y[300:], variance[300:]


def main():
    returns, true_variance = generated_returns()
    rng = np.random.default_rng(17)
    catalog = {
        "constant": vol.ConstantVarianceForecaster(mean="zero"),
        "EWMA": vol.EWMAForecaster(mean="zero"),
        "ARCH": vol.ARCHForecaster(3, mean="zero"),
        "GARCH": vol.GARCHForecaster(mean="zero"),
        "GJR-GARCH": vol.GJRGARCHForecaster(mean="zero"),
        "EGARCH": vol.EGARCHForecaster(mean="zero"),
    }
    for name, model in catalog.items():
        forecast = model.fit(returns).forecast([1, 5, 10], rng=rng)
        print(
            name,
            "conditional variance:",
            np.round(forecast.conditional_variance.ravel(), 4),
        )

    fitted = vol.GARCHForecaster(innovation="student_t", mean="zero").fit(returns)
    print(
        "One-step Student-t return interval:", fitted.forecast(1).interval(0.95).ravel()
    )
    print(
        "Annualized volatility:",
        fitted.forecast(3).volatility(periods_per_year=252).ravel(),
    )
    for name, parameters in (
        ("normal", ()),
        ("student_t", (7,)),
        ("skew_student_t", (8, -0.2)),
        ("ged", (1.5,)),
    ):
        print(
            name,
            "standardized innovation quantiles:",
            vol.innovation_distribution(name, parameters).quantile([0.01, 0.99]),
        )

    simulated = vol.GARCHForecaster(
        method="simulation", simulations=4000, mean="zero"
    ).fit(returns)
    forecast = simulated.forecast(10, rng=rng)
    assert forecast.paths is not None
    print("Return paths:", forecast.paths.shape)
    print(
        "90% future-variance interval at lead 10:",
        forecast.variance_forecast().interval(0.9)[-1, 0],
    )
    print(
        "P(last return < -2), empirical:",
        forecast.probability(lambda path: bool(path[-1, 0] < -2)),
    )

    boot = (
        vol.GARCHForecaster(method="bootstrap", mean="zero")
        .fit(returns)
        .forecast(5, rng=rng)
    )
    print("Bootstrap conditional variance:", boot.conditional_variance.ravel())
    result = vol.variance_backtest(
        catalog["GARCH"],
        returns,
        variance_proxy=true_variance,
        horizon=[1, 5],
        initial_window=1100,
        step=100,
    )
    print("Generated truth QLIKE by horizon:", result.score(by_horizon=True).ravel())
    z = np.asarray(fitted.parameters["standardized_residuals"], dtype=float)
    print(
        "Remaining ARCH LM p-value:",
        vol.volatility_diagnostics(z, lags=10)["arch_lm_pvalue"],
    )

    with TemporaryDirectory() as directory:
        path = Path(directory) / "volatility.pkl"
        fitted.save(path)
        restored = ts.load_forecaster(path)
        np.testing.assert_allclose(
            restored.forecast(4).variance(), fitted.forecast(4).variance()
        )
    updated = fitted.update([0.2, -0.3])
    print("Refit observation count:", fitted.nobs, "->", updated.nobs)


if __name__ == "__main__":
    main()
