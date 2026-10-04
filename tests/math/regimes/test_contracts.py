"""Boundary checks, immutability, extension points, and import isolation."""
import subprocess
import sys
from typing import Any
import numpy as np
import pytest
from numpy.testing import assert_allclose
from convexity.math import regimes as r, distributions as d


def invalid_boolean() -> Any:
    return 1


@pytest.mark.parametrize('action', [
    lambda: r.MarkovChain([[.9, .2], [.2, .8]]),
    lambda: r.MarkovChain([[1, 0, 0], [1, 0, 0]]),
    lambda: r.MarkovChain([[-.1, 1.1], [0, 1]]),
    lambda: r.MarkovChain([[1]]).matrix(True),
    lambda: r.MarkovChain([[1]]).sample([.9], 1, rng=np.random.default_rng()),
    lambda: r.MarkovChain([[1]]).propagate([.5, .5]),
    lambda: r.MarkovChain.fit([0, 1]),
    lambda: r.MarkovChain.fit([.5, 0]),
    lambda: r.HiddenMarkovModel(r.MarkovChain([[1]]), [d.normal()], [.5]),
    lambda: r.HiddenMarkovModel(r.MarkovChain(np.eye(2)), [d.normal(), d.poisson(1)], [.5, .5]),
    lambda: r.HiddenMarkovModel(r.MarkovChain([[1]]), [d.bernoulli(0)], [1]).filter([1]),
    lambda: r.HiddenMarkovModel(r.MarkovChain([[1]]), [d.normal()], [1]).filter([[1, 2]]),
    lambda: r.ConstantIntensity(-1), lambda: r.ConstantIntensity(np.nan),
    lambda: r.PiecewiseConstantIntensity([2, 1], [1, 2, 3]),
    lambda: r.PiecewiseConstantIntensity([1], [1]),
    lambda: r.PoissonProcess(2),  # pyright: ignore[reportArgumentType]  # Exercise runtime rejection.
    lambda: r.PoissonProcess(r.ConstantIntensity(1)).sample(1, 0, rng=np.random.default_rng()),
    lambda: r.PoissonProcess(r.ConstantIntensity(1)).log_likelihood([0], 0, 1),
    lambda: r.ExponentialHawkes(1, 1, 1), lambda: r.ExponentialHawkes(1, .1, 0),
    lambda: r.ExponentialHawkes(1, .1, 1).sample(0, 1, history=[.5], rng=np.random.default_rng()),
    lambda: r.ScheduledEvent(1, 'normal'), lambda: r.ScheduledEvent(np.inf, d.normal()),  # pyright: ignore[reportArgumentType]  # Exercise runtime rejection.
    lambda: r.ScheduledEvent(1, d.normal(), impact=3),  # pyright: ignore[reportArgumentType]  # Exercise runtime rejection.
    lambda: r.ScheduledEvent(1, d.normal(), impact=lambda surprise: 3).sample(rng=np.random.default_rng()),  # pyright: ignore[reportArgumentType]  # Exercise runtime rejection.
    lambda: r.EventSchedule([r.ScheduledEvent(1, d.normal())], (2,)).sample(0, 2, rng=np.random.default_rng()),
    lambda: r.MarkedEvents([0], [1], 0, 1), lambda: r.MarkedEvents([1], [[1], [2]], 0, 2),
    lambda: r.SwitchingRegression(regimes=1), lambda: r.SwitchingAutoregression(order=0),
    lambda: r.SwitchingRegression(switching_variance=invalid_boolean()),
    lambda: r.SwitchingRegression().fit([1, np.nan, 2]),
    lambda: r.SwitchingRegression().fit([[1, 2], [3, 4]]),
    lambda: r.RegimeResult([[1]], [[1]], [0, 0]),
    lambda: r.RegimeResult([[1]], [[1]], [0], pair_probabilities=[]),
])
def test_invalid_inputs(action):
    with pytest.raises((ValueError, TypeError)):
        action()


def test_ownership_extensibility_and_rng():
    x = np.array([[.3, .7]])
    result = r.RegimeResult(x, x, [0])
    x[:] = 100
    assert_allclose(result.filtered, [[.3, .7]])
    assert not result.filtered.flags.writeable
    class UserInference:
        def filter(self, data):
            return result
    assert r.infer_regimes([1], UserInference()) is result
    class BadInference:
        def filter(self, data):
            return None
    def invalid_inference() -> Any:
        return BadInference()
    with pytest.raises(TypeError):
        r.infer_regimes([1], invalid_inference())
    class BadProcess:
        def sample(self, start, end, *, rng):
            return [start]
    with pytest.raises(ValueError):
        r.sample_events(BadProcess(), 0, 1, rng=np.random.default_rng())  # pyright: ignore[reportArgumentType]  # Exercise runtime rejection.
    with pytest.raises(TypeError):
        r.MarkovChain([[1]]).sample([1], 1, rng=42)  # pyright: ignore[reportArgumentType]  # Exercise runtime rejection.
    with pytest.raises(TypeError):
        r.ScheduledEvent(1, d.normal()).sample(rng=None)  # pyright: ignore[reportArgumentType]  # Exercise runtime rejection.
    with pytest.raises(RuntimeError, match='max_events'):
        r.ExponentialHawkes(100, 0, 1).sample(0, 10, max_events=1, rng=np.random.default_rng(1))


def test_lazy_imports_and_exports():
    code = '''
import sys
from convexity.math import regimes as r
for name in ('scipy', 'statsmodels', 'pymc', 'torch'):
    assert name not in sys.modules, name
for name in r.__all__:
    assert hasattr(r, name), name
'''
    subprocess.run([sys.executable, '-c', code], check=True)
