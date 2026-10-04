"""Optional integrations must not reseed unrelated application calculations."""
import random
from typing import Any, cast
import numpy as np
import pytest
from convexity.math import control, reinforcement


@pytest.mark.parametrize('algorithm', ['a2c', 'ppo'])
def test_training_preserves_global_random_streams_and_repeats(algorithm):
    torch = pytest.importorskip('torch')
    pytest.importorskip('stable_baselines3')
    model = control.FiniteMDP([[[1.], [1.]]], [[0., 1.]], .5)
    python_state, numpy_state, torch_state = random.getstate(), cast(tuple[Any, ...], np.random.get_state(legacy=True)), torch.get_rng_state()
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        agents = [reinforcement.train_agent(model, algorithm=algorithm, steps=32,
                                            seed=17, n_steps=8) for _ in range(2)]
        assert random.getstate() == python_state
        current = cast(tuple[Any, ...], np.random.get_state(legacy=True))
        assert current[0] == numpy_state[0]
        np.testing.assert_array_equal(current[1], numpy_state[1])
        assert current[2:] == numpy_state[2:]
        assert torch.equal(torch.get_rng_state(), torch_state)
        assert agents[0].action(0) == agents[1].action(0)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        torch.set_num_threads(threads)
