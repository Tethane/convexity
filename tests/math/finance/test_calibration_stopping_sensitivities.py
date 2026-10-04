import numpy as np
import pytest
from convexity.math.calibration import *
from convexity.math.stopping import *
from convexity.math.sensitivities import *
from convexity.math.pricing import forward_option


def test_synthetic_vol_recovery():
    strikes = np.linspace(80, 120, 9)

    def predict(p):
        return np.array(
            [forward_option(100, k, p[0], 1, discount=np.exp(-0.03)) for k in strikes]
        )

    fit = calibrate(
        CalibrationProblem(predict, predict([0.24]), [0.4], lower=[0.01], upper=[1.0])
    )
    assert fit.success and fit.identified
    assert fit.parameters == pytest.approx([0.24], abs=1e-8)
    assert fit.covariance is not None
    assert fit.rmse < 1e-8 and fit.covariance.shape == (1, 1)
    assert fit.jacobian.shape == (9, 1)


def test_calibration_identifiability_regularization_bounds():
    x = np.arange(1, 11)

    def predict(p):
        return (p[0] + p[1]) * x

    fit = calibrate(CalibrationProblem(predict, 3 * x, [1.0, 1.0]))
    assert fit.rank == 1 and not fit.identified and fit.covariance is None
    penalized = calibrate(
        CalibrationProblem(
            predict, 3 * x, [1.0, 1.0], regularization=1, prior=[1.0, 1.0]
        )
    )
    assert penalized.covariance is None
    bounded = calibrate(
        CalibrationProblem(lambda p: p[0] * x, 3 * x, [1.0], upper=[2.0])
    )
    assert bounded.parameters == pytest.approx([2.0]) and bounded.covariance is None
    with pytest.raises(ValueError):
        CalibrationProblem(predict, 3 * x, [1.0, 1.0], weights=np.zeros(10))
    with pytest.raises(ValueError):
        calibrate(CalibrationProblem(lambda p: np.ones(3), 3 * x, [1.0]))


def test_stopping_exact_snell_and_bermudan():
    reward = np.array([[1, 1], [3, 0], [0, 6.0]])
    transitions = np.tile(np.array([[0.5, 0.5], [0.5, 0.5]]), (2, 1, 1))
    result = BackwardInduction().solve(StoppingProblem(reward, transitions, 0.9))
    np.testing.assert_allclose(result.values, [[2.565, 2.565], [3, 2.7], [0, 6]])
    assert result.exercise_region[1, 0] and not result.exercise_region[1, 1]
    european = BackwardInduction().solve(
        StoppingProblem(reward, transitions, 0.9, [False, False, True])
    )
    np.testing.assert_allclose(european.values[0], [2.43, 2.43])
    with pytest.raises(ValueError):
        StoppingProblem(reward, transitions * 2, 0.9)


def test_lsmc_independent_policy_value():
    def paths(seed, n):
        rng = np.random.default_rng(seed)
        dt = 1 / 30
        logs = np.c_[
            np.zeros(n),
            np.cumsum(
                (0.04 - 0.5 * 0.2**2) * dt
                + 0.2 * np.sqrt(dt) * rng.normal(size=(n, 30)),
                axis=1,
            ),
        ]
        s = 100 * np.exp(logs)
        return s, np.maximum(100 - s, 0)

    train, rewards = paths(12, 18000)
    policy = least_squares_policy(train, rewards, np.exp(-0.04 / 30), degree=3)
    validation, rewards = paths(31, 30000)
    estimate = policy.evaluate(validation, rewards, np.exp(-0.04 / 30))
    european = np.mean(rewards[:, -1]) * np.exp(-0.04)
    assert estimate["value"] > european + 0.2
    assert 5.5 < estimate["value"] < 6.8
    assert np.all(estimate["exercise_indices"] <= 30)
    assert estimate["standard_error"] > 0


def test_derivatives_cross_greeks_and_hedges():
    def f(x):
        return x[0] ** 2 + 3 * x[0] * x[1] + 2 * x[1] ** 2

    result = bump_sensitivities(f, [2.0, 3.0], steps=1e-3)
    np.testing.assert_allclose(result["gradient"], [13, 18], atol=1e-8)
    np.testing.assert_allclose(result["hessian"], [[2, 3], [3, 4]], atol=1e-7)
    numerical = sensitivities(lambda x: np.array([x[0] ** 2, x[0] * x[1]]), [2.0, 3.0])
    np.testing.assert_allclose(numerical.value, [[4, 0], [3, 2]], atol=1e-8)
    a = np.array([[1, 0, 1], [0, 1, 1.0]])
    fit = hedge(a, [2.0, -1.0])
    np.testing.assert_allclose(fit.residual_exposure, 0, atol=1e-12)
    np.testing.assert_allclose(a @ fit.nullspace, 0, atol=1e-12)
    bounded = hedge(a, [2.0, -1.0], lower=0)
    assert (
        np.all(bounded.holdings >= 0) and np.linalg.norm(bounded.residual_exposure) > 0
    )
    constrained = hedge(a, [2.0, -1.0], equality=[[1, 1, 1]], equality_target=[0.0])
    assert constrained.holdings.sum() == pytest.approx(0, abs=1e-10)
    with pytest.raises(RuntimeError):
        hedge(
            a,
            [2.0, -1.0],
            lower=0,
            upper=1,
            equality=[[1, 1, 1]],
            equality_target=[5.0],
        )


def test_autodiff_reuses_torch():
    pytest.importorskip("torch")
    result = sensitivities(lambda x: (x * x).sum(), [2.0, 3.0], order=2, autodiff=True)
    np.testing.assert_allclose(result.value, 2 * np.eye(2))


def test_curve_risk(discount, dividend):
    result = curve_sensitivities(
        lambda curves: 100 * curves[0].discount(2), [discount], bump=1e-4
    )
    assert result["gradient"][0] == pytest.approx(-200 * np.exp(-0.1), rel=1e-7)
    assert result["hessian"][0, 0] == pytest.approx(400 * np.exp(-0.1), rel=1e-7)


def test_quantlib_native_calibration(discount, dividend, valuation):
    import QuantLib as ql
    from convexity.math._finance import evaluation_date
    from convexity.math.pricing import HestonModel

    with evaluation_date(valuation):
        native = HestonModel(
            100, discount, dividend, 0.04, 2.0, 0.04, 0.3, -0.6
        ).to_quantlib()
        helpers = []
        for months in (6, 12):
            for strike in (90.0, 100.0, 110.0):
                quote = ql.SimpleQuote(0.2)
                helper = ql.HestonModelHelper(
                    ql.Period(months, ql.Months),
                    ql.NullCalendar(),
                    100.0,
                    strike,
                    ql.QuoteHandle(quote),
                    discount.handle,
                    dividend.handle,
                )
                helper.setPricingEngine(ql.AnalyticHestonEngine(native))
                synthetic = helper.modelValue()
                quote.setValue(
                    helper.impliedVolatility(synthetic, 1e-10, 1000, 0.001, 2.0)
                )
                helpers.append(helper)
        native.setParams([0.04, 2.0, 0.3, -0.6, 0.08])
        result = quantlib_calibrate(
            native, helpers, valuation, fixed_parameters=[True, True, True, True, False]
        )
        assert result["parameters"][-1] == pytest.approx(0.04, abs=1e-7)
        assert np.max(np.abs(result["residuals"])) < 1e-6
