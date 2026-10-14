"""Public generated-data estimation through measure change, pricing and control."""

import numpy as np
from numpy.testing import assert_allclose
from convexity.math import (
    estimation,
    distributions,
    measures,
    processes,
    simulation,
    pricing,
    risk,
    optimization,
    control,
)


def test_estimated_physical_model_to_explicit_pricing_and_hedge_decision():
    rng = np.random.default_rng(42017)
    returns = distributions.normal(0.06, 0.2).sample(20000, rng=rng)
    estimate = estimation.linear_least_squares(np.ones((len(returns), 1)), returns)
    log_drift = float(estimate.parameters[0])
    sigma = float(np.std(returns))
    assert abs(log_drift - 0.06) < 5 * 0.2 / np.sqrt(len(returns))
    assert abs(sigma - 0.2) < 0.005
    pair = measures.gbm_pair(log_drift + 0.5 * sigma**2, 0.03, 0.0, sigma)
    assert isinstance(pair.physical.model, processes.GeometricBrownianMotion)
    assert pair.physical.measure == "P" and pair.pricing.measure == "Q"
    assert_allclose(
        pair.market_price_of_risk(0.0, 100.0), [(pair.physical.model.mu - 0.03) / sigma]
    )
    # Q is supplied explicitly. It is never inferred from fitted P coefficients.
    paths = simulation.simulate(
        pair.pricing.model,
        100.0,
        [0.0, 0.5, 1.0],
        draws=12000,
        rng=np.random.default_rng(431),
    )
    forecast = paths.to_forecast()
    assert_allclose(forecast.mean()[:, 0], paths.mean())
    assert_allclose(forecast.covariance(), paths.covariance())
    price = pricing.scenario_price(paths, lambda p: p[-1], pair.pricing)
    # Discounted stock is a Q martingale in this zero-dividend model.
    assert price.error_estimate is not None
    assert abs(price.value - 100.0) < 5 * price.error_estimate
    terminal = paths.values[:, -1]
    loss = 100 * np.exp(0.03) - terminal
    expected_es = np.sort(loss)[-600:].mean()  # exactly 5% of equal masses
    assert_allclose(
        risk.expected_shortfall(loss, 0.95, weights=paths.weights), expected_es
    )
    # Projection hedge: minimize E[(q*S_T - (2*S_T+3))^2]/2.
    liability = 2 * terminal + 3
    moment = paths.weights @ terminal**2
    cross = paths.weights @ (terminal * liability)
    solution = optimization.solve(optimization.QuadraticProgram([-cross], [[moment]]))
    assert solution.solution is not None
    assert_allclose(solution.solution, [cross / moment], rtol=1e-5)
    # Dynamic control on the estimated scalar transition, tested via Bellman
    # identity rather than agreement with the provider's Riccati solver.
    growth = np.exp(log_drift + 0.5 * sigma**2)
    lqr = control.lqr([[growth]], [[1.0]], [[1.0]], [[1.0]])
    expected_value = (growth**2 + np.sqrt(growth**4 + 4)) / 2
    assert_allclose(lqr.riccati, [[expected_value]], rtol=1e-10)
