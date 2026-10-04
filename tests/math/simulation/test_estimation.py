import numpy as np
import pytest
from numpy.testing import assert_allclose
from convexity.math import simulation as s


def test_iid_pair_and_unknown_standard_errors():
    x = np.array([1., 2., 4., 3.])
    result = s.estimate(x)
    assert_allclose(result.value, 2.5)
    assert result.standard_error is not None
    assert_allclose(result.standard_error, x.std(ddof=1)/2)
    paired = s.estimate(x, sampling='antithetic')
    assert paired.independent_units == 2
    assert paired.standard_error == 0
    assert s.estimate([3.]).standard_error is None
    for sampling in ('sobol', 'halton', 'unspecified'):
        assert s.estimate(x, sampling=sampling).standard_error is None
    with pytest.raises(ValueError):
        s.estimate([1, 2, 3], sampling='antithetic')


def test_antithetic_variance_reduction_and_known_control_identity():
    z = s.gaussian_innovations(2000, [[1]], antithetic=True, rng=np.random.default_rng(3))[:, 0]
    ordinary = s.estimate(np.exp(.3*z))
    paired = s.estimate(np.exp(.3*z), sampling='antithetic')
    assert ordinary.standard_error is not None and paired.standard_error is not None
    assert paired.standard_error < ordinary.standard_error/3
    adjusted = s.control_variates(5+2*z, z[:, None], [0], [2])
    assert_allclose(adjusted, 5)
    result = s.estimate(adjusted, sampling='antithetic')
    assert result.standard_error is not None
    assert result.standard_error < 1e-15


def test_weighted_scenarios_do_not_claim_iid_error_or_evaluate_zero_mass():
    scenarios = s.PathScenarios([0], [[1.], [3.], [-1.]], [1, 3, 0], sampling='iid')
    def function(x):
        assert x[0] > 0
        return x[0]**2
    estimate = s.estimate_scenarios(scenarios, function)
    assert estimate.value == 7
    assert estimate.standard_error is None
    assert estimate.independent_units == 0


@pytest.mark.parametrize('method', ['sobol', 'halton'])
def test_randomized_qmc_independent_scramble_error(method):
    result = s.qmc_expectation(lambda z: np.exp(.2*z[:, 0]), 2, draws=1024,
                               replicates=8, rng=np.random.default_rng(22), method=method)
    assert result.independent_units == 8
    assert abs(result.value-np.exp(.02)) < .001
    assert result.standard_error is not None
    assert 0 < result.standard_error < .001
    replay = s.qmc_expectation(lambda z: np.exp(.2*z[:, 0]), 2, draws=1024,
                               replicates=8, rng=np.random.default_rng(22), method=method)
    assert result.value == replay.value
    with pytest.raises(ValueError):
        s.qmc_expectation(lambda z: z.mean(), 2, draws=8, replicates=2, rng=np.random.default_rng(1))


def test_quantlib_process_backend():
    from convexity.math import processes as p
    pytest.importorskip('QuantLib')
    result = s.simulate(p.Heston(.02, 2, .04, .4, -.7), [100, .04], [0, .2, 1],
                        draws=20, method='qe', substeps=4, rng=np.random.default_rng(2))
    assert result.backend == 'QuantLib' and not result.exact
    assert np.all(result.values[:, :, 0] > 0) and np.all(result.values[:, :, 1] >= 0)


def test_sdeint_process_backend():
    from convexity.math import processes as p
    pytest.importorskip('sdeint')
    model = p.ItoDiffusion(lambda t, x: -x, lambda t, x: [0])
    result = s.simulate(model, 1, [0, 1], method='sri2', substeps=32, rng=np.random.default_rng(1))
    assert result.backend == 'sdeint' and not result.exact
    assert_allclose(result.values[0, -1], np.exp(-1), atol=.0001)
