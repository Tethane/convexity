import numpy as np
import pytest
from convexity.math import risk, utility
from convexity.math.simulation import PathScenarios


def test_fractional_atom_es_matches_hinge_minimum():
    x, w = np.array([-3.0, 1.0, 7.0, 9.0]), np.array([0.3, 0.4, 0.2, 0.1])
    for level in (0.0, 0.3, 0.5, 0.8, 0.95):
        es = risk.expected_shortfall(x, level, weights=w)
        hinge = min(z + w @ np.maximum(x - z, 0) / (1 - level) for z in x)
        assert es == pytest.approx(hinge)
    assert risk.value_at_risk(x, 0.5, weights=w) == 1
    assert risk.expected_shortfall(x, 0.5, weights=w) == pytest.approx(5.0)


def test_zero_mass_quantile_and_extreme_weights():
    assert risk.value_at_risk([-100.0, 1.0, 3.0, 100.0], 0, weights=[0, 1, 1, 0]) == 1
    assert risk.value_at_risk([-100.0, 1.0, 3.0, 100.0], 1, weights=[0, 1, 1, 0]) == 3
    law = risk.LossScenarios([1, 3], [1e308, 1e308])
    assert risk.expected_shortfall(law, 0) == 2
    with pytest.raises(ValueError):
        risk.expected_shortfall(law, 1)  # pyright: ignore[reportArgumentType]  # Exercise runtime rejection.


def test_es_translation_homogeneity_subadditivity():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=(2, 100))
    es = risk.expected_shortfall
    assert es(x + 4) == pytest.approx(es(x) + 4)
    assert es(3 * x) == pytest.approx(3 * es(x))
    assert es(x + y) <= es(x) + es(y) + 1e-12


def test_euler_contributions_and_finite_difference():
    h = np.array([2.0, -1.0])
    c = np.array([[2.0, 0.5], [0.5, 1.0]])
    r = risk.volatility_contributions(h, c)
    assert r.component.sum() == pytest.approx(r.value)
    for i in range(2):
        d = np.eye(2)[i] * 1e-5
        finite = (
            np.sqrt((h + d) @ c @ (h + d)) - np.sqrt((h - d) @ c @ (h - d))
        ) / 2e-5
        assert finite == pytest.approx(r.marginal[i])
    x = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 1.0]])
    r = risk.es_contributions(h, x, 0.5)
    assert r.component.sum() == pytest.approx(risk.expected_shortfall(x @ h, 0.5))
    with pytest.raises(ValueError):
        risk.volatility_contributions([0, 0], c)


def test_paths_stress_margin_and_downside():
    x = np.array([[100.0, 120.0, 90.0, 110.0], [100.0, 110.0, 105.0, 130.0]])
    paths = PathScenarios([0.0, 1.0, 2.0, 3.0], x, [0.2, 0.8])
    np.testing.assert_allclose(risk.maximum_drawdown(x), [0.25, 5 / 110])
    assert risk.breach_probability(paths, 90) == pytest.approx(0.2)
    assert risk.breach_probability(paths, 130, direction="above") == pytest.approx(0.8)
    assert risk.downside_moment([-2.0, 1.0], weights=[0.25, 0.75]) == 1
    assert risk.scenario_margin([-10, 5], addon=2, multiplier=3) == 17
    np.testing.assert_allclose(risk.stress([1, 2], shift=1, scale=2), [3, 5])
    with pytest.raises(ValueError):
        risk.drawdown([0, 1])


@pytest.mark.parametrize(
    "u",
    [
        utility.LinearUtility(),
        utility.LogUtility(),
        utility.PowerUtility(0.5),
        utility.PowerUtility(1),
        utility.PowerUtility(3),
        utility.ExponentialUtility(0.3),
        utility.SmoothUtility(0.7, 2),
    ],
)
def test_utility_inverse_ce_and_jensen(u):
    x = np.array([0.5, 1.0, 2.0, 5.0])
    np.testing.assert_allclose(u.inverse(u(x)), x, rtol=1e-11)
    ce = utility.certainty_equivalent(x, u)
    assert x.min() <= ce <= x.mean() + 1e-10
    assert utility.certainty_equivalent([3.0, 3.0], u) == pytest.approx(3.0)


def test_options_utility_negative_wealth_and_exponential_stability():
    u = utility.SmoothUtility(0.6, 3.0)
    x = np.linspace(-100, 100, 1001)
    np.testing.assert_allclose(u.inverse(u(x)), x, atol=1e-12)
    assert np.all(np.diff(u(x)) > 0)
    assert np.all(np.diff(u(x), n=2) <= 1e-10)
    ce = utility.certainty_equivalent(
        [-10000.0, -10001.0], utility.ExponentialUtility(10.0)
    )
    assert np.isfinite(ce) and -10001 <= ce <= -10000
    assert utility.expected_utility(
        [-1.0, 2.0], utility.LogUtility(), weights=[0, 1]
    ) == pytest.approx(np.log(2))


def test_preferences_are_explicit():
    assert utility.preference_penalty([0.0, 0.0], switching=10) == 0
    assert (
        utility.preference_penalty([2.0, -1.0], turnover=0.5, switching=3, scale=[2, 1])
        == 5.5
    )
    assert utility.mean_variance([1.0, 3.0], 2.0) == pytest.approx(1.0)
    assert utility.mean_cvar([1.0, 3.0], 1.0, 0.5) == pytest.approx(3.0)
