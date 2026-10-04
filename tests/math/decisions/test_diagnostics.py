import numpy as np
import pytest
from convexity.math import diagnostics as d
from convexity.math.simulation import PathScenarios


def test_weighted_scores_against_pairwise_definitions():
    x, w, y = np.array([-2.0, 1.0, 3.0]), np.array([0.1, 0.3, 0.6]), 2.0
    exact = w @ abs(x - y) - 0.5 * np.sum(
        w[:, None] * w[None, :] * abs(x[:, None] - x[None, :])
    )
    assert d.crps(x, y, weights=w) == pytest.approx(exact)
    assert d.energy_score(x[:, None], [y], weights=w) == pytest.approx(exact)
    assert d.crps([2.0, 2.0], 1.0) == 1.0
    assert d.quantile_score([1, 2], [2, 1], 0.5) == 0.5
    assert d.log_score([-np.inf, -2], weights=[0, 1]) == 2
    assert d.log_score([-np.inf]) == np.inf


def test_joint_path_scores_detect_wrong_dependence():
    good = np.array([[-1.0, -1.0], [1.0, 1.0]])
    bad = np.array([[-1.0, 1.0], [1.0, -1.0]])
    a = sum(d.energy_score(good, y) for y in good)
    b = sum(d.energy_score(bad, y) for y in good)
    assert a < b
    assert d.variogram_score(good, [1.0, 1.0]) == 0
    assert d.variogram_score(bad, [1.0, 1.0]) > 0
    paths = PathScenarios([0.0, 1.0], good)
    assert d.energy_score(paths, [1.0, 1.0]) == pytest.approx(
        d.energy_score(good, [1.0, 1.0])
    )


def test_pit_and_coverage():
    rng = np.random.default_rng(31)
    actual = rng.binomial(1, 0.3, size=20000)
    hi = np.where(actual == 0, 0.7, 1.0)
    lo = np.where(actual == 0, 0.0, 0.7)
    pit = d.pit(hi, lower_cdf=lo, rng=rng)
    assert pit.mean() == pytest.approx(0.5, abs=0.006)
    assert np.mean(pit < 0.1) == pytest.approx(0.1, abs=0.006)
    assert d.interval_coverage([0, 1, 3], [-1, 0, 0], [1, 2, 2]) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        d.pit([0.4], lower_cdf=[0.5], rng=rng)


def test_residual_and_stationarity_generated_samples():
    rng = np.random.default_rng(13)
    iid = rng.normal(size=1500)
    ar = np.zeros(1500)
    for t in range(1, len(ar)):
        ar[t] = 0.8 * ar[t - 1] + iid[t]
    assert d.residual_test(ar).pvalue[-1] < 1e-6
    assert d.stationarity_test(ar, lags=3).pvalue < 0.01
    walk = np.cumsum(rng.normal(size=1500))
    assert d.cointegration_test(walk + 0.1 * iid, walk, lags=1).pvalue < 0.01
    result = d.residual_test(iid, method="jarque_bera")
    assert 0 <= result.pvalue <= 1
    with pytest.warns(Warning):
        kpss = d.stationarity_test(iid, method="kpss")
    assert kpss.pvalue >= 0.05


def test_hac_matches_direct_bartlett_and_model_comparison():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(200, 2))
    z = x - x.mean(0)
    exact = z.T @ z
    for lag in range(1, 5):
        gamma = z[lag:].T @ z[:-lag]
        exact += (1 - lag / 5) * (gamma + gamma.T)
    np.testing.assert_allclose(
        d.hac_mean_covariance(x, lags=4), exact / len(x) ** 2, atol=1e-15
    )
    result = d.compare_scores(x[:, 0], x[:, 0] + 1 + 0.1 * x[:, 1], lags=3)
    assert result.statistic < 0 and result.pvalue < 0.001
    with pytest.raises(ValueError):
        d.compare_scores([1, 2, 3], [1, 2, 3], lags=0)


def test_block_bootstrap_pairing_and_multiplicity():
    rng = np.random.default_rng(12)
    x = rng.normal(size=100)
    paired = np.column_stack([x, 2 * x])
    r = d.block_bootstrap(
        lambda a: a.mean(0), paired, block_length=5, draws=100, rng=rng
    )
    np.testing.assert_allclose(r.samples[:, 1], 2 * r.samples[:, 0])
    assert not r.samples.flags.writeable
    assert np.all(r.lower <= r.upper)
    np.testing.assert_allclose(
        d.adjust_pvalues([0.01, 0.04, 0.03], method="holm"), [0.03, 0.06, 0.06]
    )
    np.testing.assert_allclose(
        d.adjust_pvalues([0.01, 0.04, 0.03], method="bh"), [0.03, 0.04, 0.04]
    )
    with pytest.raises(ValueError):
        d.block_bootstrap(np.mean, x, block_length=101, draws=10, rng=rng)
