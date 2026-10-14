"""Recovery and an analytic reverse-diffusion reference, beyond shape tests."""

import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose

from convexity.math import forecasting as f

torch = pytest.importorskip("torch", exc_type=ModuleNotFoundError)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize(
    "factory",
    [
        f.FlowForecaster,
        f.VAEForecaster,
        f.AutoregressiveForecaster,
        f.ScoreForecaster,
        f.DiffusionForecaster,
    ],
)
def test_generative_training_recovers_held_out_periodic_cycle(factory):
    cycle = np.array([0.0, 1.0, 0.0, -1.0])
    fitted = factory(
        context_length=4,
        max_horizon=4,
        hidden_size=32,
        epochs=100,
        samples=2000,
        seed=123,
    ).fit(np.tile(cycle, 50))
    forecast = fitted.forecast(4, rng=np.random.default_rng(42))
    # Both zero and last-observation baselines have RMSE > .7 here. Learning
    # must recover the next cycle, not merely reduce the training objective.
    assert np.sqrt(np.mean((forecast.point[:, 0] - cycle) ** 2)) < 0.15
    assert fitted.diagnostics.loss_history[-1] < fitted.diagnostics.loss_history[0]
    # This is a deterministic conditional target, so enormous sample spread
    # is also a failure. This bound is not a probabilistic calibration claim.
    assert np.sqrt(np.mean((forecast.paths[:, :, 0] - cycle) ** 2)) < 0.5


class KnownPointDenoiser(torch.nn.Module):
    """Exact noise or velocity for a known deterministic clean path."""

    def __init__(self, alpha, clean, prediction_type):
        super().__init__()
        self.alpha = alpha
        self.clean = torch.tensor(clean, dtype=torch.float32)
        self.prediction_type = prediction_type

    def forward(self, noisy, time, condition):
        index = torch.round(time * (len(self.alpha) - 1)).long().cpu().numpy()
        alpha = torch.tensor(self.alpha[index], dtype=noisy.dtype)
        noise = (noisy - alpha.sqrt() * self.clean) / (1 - alpha).sqrt()
        if self.prediction_type == "epsilon":
            return noise
        return alpha.sqrt() * noise - (1 - alpha).sqrt() * self.clean


@pytest.mark.parametrize("steps", [2, 7, 50])
@pytest.mark.parametrize("prediction_type", ["epsilon", "v_prediction"])
def test_reverse_diffusion_with_exact_denoiser_returns_known_path(
    steps, prediction_type
):
    from convexity.math.forecasting._backends.diffusion import (
        DiffusionAdapter,
        schedule,
    )

    clean = np.array([0.4, -0.7])
    spec = f.DiffusionForecaster(
        context_length=2, max_horizon=2, diffusion_steps=steps, samples=100
    )
    _, alpha = schedule(steps)
    adapter = DiffusionAdapter(
        KnownPointDenoiser(alpha, clean, prediction_type),
        f.TimeSeries([1.0, 2.0, 1.0, 2.0]),
        spec,
        np.array([3.0]),
        np.array([2.0]),
        prediction_type,
    )
    if prediction_type == "epsilon":
        # Old artifacts have no prediction_type field. Their weights must not
        # silently acquire the interpretation used by newly trained models.
        object.__delattr__(adapter, "prediction_type")
        adapter = pickle.loads(pickle.dumps(adapter))
        assert adapter.prediction_type == "epsilon"
    result = adapter.forecast(2, None, rng=np.random.default_rng(20))
    assert result.paths is not None
    assert_allclose(
        result.paths[:, :, 0], np.broadcast_to(3 + 2 * clean, (100, 2)), atol=2e-5
    )
