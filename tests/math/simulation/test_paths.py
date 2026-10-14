import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from convexity.math import simulation as s, processes as p, distributions as d


def test_weighted_joint_law_callbacks_resampling_and_forecast():
    raw = np.array(
        [[[0.0, 1.0], [2.0, 4.0]], [[0.0, 1.0], [6.0, 8.0]], [[9.0, 9.0], [9.0, 9.0]]]
    )
    paths = s.PathScenarios([0.0, 1.0], raw, [1.0, 3.0, 0.0])
    raw[:] = -100
    assert_allclose(paths.mean(), [[0, 1], [5, 7]])
    assert_allclose(paths.covariance()[2:, 2:], [[3, 3], [3, 3]])
    assert_allclose(paths.effective_size, 1 / 0.625)
    assert paths.probability(lambda x: x[1, 0] > 3) == 0.75
    assert_allclose(paths.expect(lambda x: x[-1] - x[0]), [5, 6])
    assert_allclose(paths.map(lambda x: x[-1, 0]).quantile([0.25, 0.26]), [2, 6])
    assert_allclose(
        paths.distribution().log_density(paths.values[:2]), np.log([0.25, 0.75])
    )
    assert_allclose(paths.distribution().marginal(1).mean(), [5, 7])
    forecast = paths.to_forecast()
    assert_allclose(forecast.mean(), paths.mean())
    assert_allclose(forecast.covariance(), paths.covariance())
    draw = paths.resample(1000, rng=np.random.default_rng(1))
    assert_array_equal(draw.values[:, 1, 1], draw.values[:, 1, 0] + 2)
    assert abs(draw.mean()[1, 0] - 5) < 0.15
    assert_allclose(paths.reweight([3, 1, 1]).weights, [0.5, 0.5, 0])
    with pytest.raises(ValueError):
        paths.values[0, 0, 0] = 1
    with pytest.raises(ValueError):
        paths.reweight([0, 0, 1])


def test_joint_gaussian_brownian_covariance_and_initial_uncertainty():
    times = [2.0, 2.25, 3.0]
    law = s.gaussian_path_distribution(
        1.0, times, lambda t, dt: (1.0, 0.2 * dt, 0.3**2 * dt), initial_covariance=0.4
    )
    elapsed = np.array(times) - times[0]
    assert_allclose(law.mean(), 1 + 0.2 * elapsed)
    expected = 0.4 + 0.09 * np.minimum.outer(elapsed, elapsed)
    assert_allclose(law.covariance(), expected)
    values = law.sample(20000, rng=np.random.default_rng(3))
    se = np.sqrt(
        (expected**2 + np.outer(np.diag(expected), np.diag(expected)))
        / (len(values) - 1)
    )
    assert np.all(abs(np.cov(values.T) - expected) < 5 * se)
    assert law.sample((2, 3), rng=np.random.default_rng(0)).shape == (2, 3, 3)
    assert law.sample(0, rng=np.random.default_rng(0)).shape == (0, 3)
    assert law.log_density(values[:2]).shape == (2,)
    with pytest.raises(ValueError):
        law.log_density([1, 2])


def test_vector_gaussian_joint_covariance_obeys_ou_semigroup():
    model = p.VectorOrnsteinUhlenbeck(
        [[0.8, 0.2], [0.0, 0.4]], [1.0, -1.0], [[0.4], [0.2]]
    )
    law = s.gaussian_path_distribution(
        [0.0, 0.0], [0.0, 0.3, 1.0], lambda t, dt: model.coefficients(dt)
    )
    kernel = model.transition([0.0, 0.0], 1.0)
    assert_allclose(law.mean()[-1], kernel.mean())
    assert_allclose(law.covariance()[-2:, -2:], kernel.covariance())
    assert_allclose(
        law.covariance()[2:4, 4:],
        model.coefficients(0.3)[2] @ model.coefficients(0.7)[0].T,
    )
    assert_allclose(law.sample(3, rng=np.random.default_rng(1))[:, 0], 0, atol=1e-10)


def test_sequential_exact_and_numerical_methods_retain_provenance():
    x = s.simulate(
        p.BrownianMotion(), 0.0, [0, 0.2, 1], draws=4000, rng=np.random.default_rng(1)
    )
    assert x.exact and x.sampling == "iid" and x.method == "exact"
    assert_allclose(x.covariance()[1:, 1:], [[0.2, 0.2], [0.2, 1]], atol=0.025)
    cir = s.simulate(
        p.CIR(2.0, 0.04, 0.5), 0.0, [0, 0.5, 1], draws=8, rng=np.random.default_rng(1)
    )
    assert cir.exact and np.all(cir.values >= 0)
    # Existing model methods are explicitly passed, never silently substituted.
    with pytest.raises(ValueError):
        s.simulate(
            p.Heston(0.02, 2, 0.04, 0.4, -0.7),
            [100, 0.04],
            [0, 1],
            rng=np.random.default_rng(0),
        )


def test_conditional_observations_preserve_latent_observation_dependence():
    latent = s.PathScenarios(
        [0, 1], np.random.default_rng(2).normal(size=(4000, 2)), sampling="iid"
    )
    result = s.simulate_observations(
        latent, lambda t, x: d.normal(2 * x + t, 0.3), rng=np.random.default_rng(3)
    )
    residual = result.observations.values - 2 * latent.values - np.array([0, 1])
    assert abs(residual.mean()) < 0.015
    assert_allclose(residual.var(), 0.09, atol=0.004)
    joint = result.joint()
    assert_allclose(joint.values[:, :, 0], latent.values)
    assert_allclose(joint.covariance()[0, 1], 2 * latent.values[:, 0].var(), atol=0.02)
    assert result.observations.sampling == "iid"
    weighted = s.PathScenarios([0], [[1], [2]], [1, 3])
    observed = s.simulate_observations(
        weighted, lambda t, x: d.empirical([x]), rng=np.random.default_rng(1)
    )
    assert_allclose(observed.joint().weights, [0.25, 0.75])


def test_structural_sampler_and_contract_failures():
    class Sampler:
        def sample(self, initial, times, *, draws, rng):
            return s.PathScenarios(times, np.full((draws, len(times)), initial))

    assert (
        s.sample_paths(
            Sampler(), 3, [0, 1], draws=2, rng=np.random.default_rng(1)
        ).mean()[1]
        == 3
    )
    for times, values, weights in [
        ([0, 0], [[1, 2]], None),
        ([0], [[1], [2]], [0, 0]),
        ([0], [[1]], [-1]),
        ([0], [[np.inf]], None),
        ([0], np.empty((0, 1)), None),
    ]:
        with pytest.raises(ValueError):
            s.PathScenarios(times, values, weights)
    with pytest.raises(ValueError):
        s.PathDistribution([0, 1], (), d.normal())
    with pytest.raises(ValueError):
        s.gaussian_path_distribution(0, [0, 1], lambda t, dt: (1, 0, -1))
    with pytest.raises(TypeError):
        s.simulate_observations(
            s.PathScenarios([0], [[1]]),
            lambda t, x: 1,  # pyright: ignore[reportArgumentType] -- deliberate runtime rejection
            rng=np.random.default_rng(1),
        )


def test_user_sampler_output_and_observation_shape_checks():
    class BadSampler:
        def sample(self, initial, times, *, draws, rng):
            return s.PathScenarios(times, np.zeros((draws + 1, len(times))))

    with pytest.raises(ValueError, match="draws and times"):
        s.sample_paths(BadSampler(), 0, [0, 1], rng=np.random.default_rng(1))
    latent = s.PathScenarios([0, 1], [[1, 2]])
    with pytest.raises(ValueError, match="event_shape"):
        s.simulate_observations(
            latent,
            lambda t, x: d.normal() if t == 0 else d.empirical([[1, 2]]),
            rng=np.random.default_rng(1),
        )


def test_observation_weights_survive_renormalization_roundoff():
    rng = np.random.default_rng(2)
    latent = s.PathScenarios([0], np.ones((10, 1)), rng.uniform(size=10))
    result = s.simulate_observations(latent, lambda t, x: d.empirical([x]), rng=rng)
    assert_allclose(result.joint().weights, latent.weights, rtol=1e-14)
