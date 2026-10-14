import numpy as np
import pytest
from numpy.testing import assert_allclose

from convexity.math import forecasting as f

# Import failures other than an absent optional package must surface as failures.
torch = pytest.importorskip("torch", exc_type=ModuleNotFoundError)


@pytest.fixture(autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize(
    "architecture", ["mlp", "cnn", "rnn", "gru", "lstm", "transformer"]
)
def test_sequence_architectures_train(architecture):
    y = np.sin(np.arange(60.0) / 5)
    spec = f.TorchForecaster(
        architecture=architecture,
        context_length=6,
        max_horizon=3,
        hidden_size=8,
        epochs=3,
        batch_size=12,
        validation_size=10,
    )
    fitted = spec.fit(y)
    out = fitted.forecast([1, 3])
    assert out.shape == (2, 1)
    assert np.isfinite(out.point).all()
    assert fitted.diagnostics.loss_history is not None
    assert fitted.diagnostics.loss_history.shape == (3,)
    assert fitted.diagnostics.validation_history is not None
    assert fitted.diagnostics.validation_history.shape == (3,)


def test_neural_recovery_reproducibility_and_checkpoint(tmp_path):
    y = np.tile([0.0, 1.0, 0.0, -1.0], 20)
    spec = f.TorchForecaster(
        context_length=4,
        max_horizon=4,
        hidden_size=16,
        epochs=50,
        batch_size=32,
        learning_rate=0.03,
        seed=123,
    )
    before = torch.random.get_rng_state().clone()
    fitted = spec.fit(y)
    assert torch.equal(before, torch.random.get_rng_state())
    assert_allclose(fitted.predict(4)[:, 0], [0, 1, 0, -1], atol=0.1)
    repeated = spec.fit(y)
    assert_allclose(fitted.predict(4), repeated.predict(4), atol=0, rtol=0)
    path = tmp_path / "torch.pkl"
    fitted.save(path)
    assert_allclose(
        f.load_forecaster(path).predict(4), fitted.predict(4), atol=0, rtol=0
    )
    assert fitted.diagnostics.loss_history is not None
    assert fitted.diagnostics.loss_history[-1] < fitted.diagnostics.loss_history[0] / 10


@pytest.mark.parametrize("head", ["gaussian", "quantile"])
def test_neural_probabilistic_heads_and_covariates(head):
    rng = np.random.default_rng(3)
    y = rng.normal(size=(55, 2))
    y[10, 0] = np.nan
    cov = f.Covariates(
        known_future=rng.normal(size=(55, 1)),
        observed_past=rng.normal(size=(55, 2)),
        static=[1],
    )
    spec = f.TorchForecaster(
        context_length=5,
        max_horizon=3,
        hidden_size=8,
        head=head,
        epochs=3,
        missing="mask",
    )
    fitted = spec.fit(y, covariates=cov)
    result = fitted.forecast(3, covariates=np.ones((3, 1)))
    assert result.shape == (3, 2)
    assert np.all(result.quantile(0.1) <= result.quantile(0.9))
    if head == "gaussian":
        assert np.all(result.variance() > 0)
        assert result.sample(5, rng=rng).shape == (5, 3, 2)
        assert_allclose(result.covariance(), np.diag(result.variance().ravel()))
    else:
        assert "sample" not in result.capabilities
    with pytest.raises(ValueError):
        fitted.forecast(2, covariates=np.ones((2, 1)))


class UserSequence(torch.nn.Module):
    def __init__(self, schema):
        super().__init__()
        self.linear = torch.nn.Linear(schema.targets, schema.horizon * schema.targets)
        self.schema = schema

    def forward(self, batch):
        return self.linear(batch["history"][:, -1]).reshape(
            -1, self.schema.horizon, self.schema.targets, 1
        )


def user_loss(raw, target, mask):
    return (((raw[..., 0] - target) ** 2) * mask).sum() / mask.sum()


def user_decoder(raw, location, scale, *, rng):
    point = raw[0, ..., 0].detach().cpu().numpy() * scale + location
    return f.ForecastResult(point)


def test_user_module_and_custom_head():
    spec = f.TorchForecaster(
        module_factory=UserSequence,
        context_length=4,
        max_horizon=2,
        epochs=2,
        head="custom",
        loss=user_loss,
        decoder=user_decoder,
    )
    out = spec.fit(np.arange(25.0)).forecast(2)
    assert out.shape == (2, 1)
    assert out.capabilities == {"point"}


def test_validation_scaling_does_not_see_holdout():
    y = np.arange(40.0)
    spec = f.TorchForecaster(
        context_length=4, max_horizon=2, epochs=1, validation_size=10
    )
    a = spec.fit(y)
    y[30:] = 1e6
    b = spec.fit(y)
    assert_allclose(
        np.asarray(a.parameters["location"], dtype=float),
        np.asarray(b.parameters["location"], dtype=float),
    )
    assert_allclose(
        np.asarray(a.parameters["scale"], dtype=float),
        np.asarray(b.parameters["scale"], dtype=float),
    )
    assert (
        a.diagnostics.loss_history is not None
        and b.diagnostics.loss_history is not None
    )
    assert_allclose(a.diagnostics.loss_history, b.diagnostics.loss_history)


@pytest.mark.parametrize("head", ["point", "lognormal"])
def test_incremental_neural_context_preserves_fit(head):
    from dataclasses import replace
    from convexity.math.forecasting._backends.torch import TorchAdapter

    values = np.exp(np.sin(np.arange(30.0) / 5))
    spec = f.TorchForecaster(
        architecture="transformer",
        head=head,
        context_length=4,
        max_horizon=2,
        hidden_size=8,
        epochs=2,
    )
    fitted = spec.fit(values)
    assert isinstance(fitted.adapter, TorchAdapter)
    weights = {
        name: value.clone()
        for name, value in fitted.adapter.module.state_dict().items()
    }
    before = fitted.predict(2).copy()

    updated = fitted.update([2.0, 3.0], mode="incremental")
    assert isinstance(updated.adapter, TorchAdapter)

    assert fitted.nobs == 30
    assert updated.nobs == 32
    assert_allclose(fitted.predict(2), before)
    assert_allclose(updated.adapter.location, fitted.adapter.location)
    assert_allclose(updated.adapter.scale, fitted.adapter.scale)
    for name, value in updated.adapter.module.state_dict().items():
        assert torch.equal(value, weights[name])

    # A forward pass on the extended context must agree, including the log
    # transform used by positive-valued heads. No optimizer should run here.
    expected_values = np.r_[values, 2.0, 3.0]
    if head == "lognormal":
        expected_values = np.log(expected_values)
    reference = replace(fitted.adapter, history=f.TimeSeries(expected_values))
    assert_allclose(updated.predict(2), reference.forecast(2, None).point)

    if head == "lognormal":
        with pytest.raises(ValueError, match="strictly positive"):
            updated.update([0.0], mode="incremental")


def test_diffusion_joint_samples_seed_and_artifact(tmp_path):
    spec = f.DiffusionForecaster(
        context_length=4,
        max_horizon=2,
        hidden_size=8,
        diffusion_steps=6,
        epochs=4,
        samples=20,
    )
    fitted = spec.fit(np.sin(np.arange(45.0) / 3))
    before = torch.random.get_rng_state().clone()
    a = fitted.forecast(2, rng=np.random.default_rng(5))
    b = fitted.forecast(2, rng=np.random.default_rng(5))
    assert torch.equal(before, torch.random.get_rng_state())
    assert a.paths is not None
    assert a.paths.shape == (20, 2, 1)
    assert a.paths is not None and b.paths is not None
    assert_allclose(a.paths, b.paths, atol=0, rtol=0)
    assert np.all(a.variance() > 0)
    path = tmp_path / "diffusion.pkl"
    fitted.save(path)
    assert a.paths is not None
    assert a.paths is not None
    restored = f.load_forecaster(path).forecast(2, rng=np.random.default_rng(5))
    assert restored.paths is not None
    assert_allclose(restored.paths, a.paths)
    with pytest.raises(TypeError):
        fitted.forecast(2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is optional")
def test_cuda_forecaster_training_and_rng_isolation():
    before = torch.cuda.get_rng_state().clone()
    spec = f.TorchForecaster(context_length=3, max_horizon=2, epochs=2, device="cuda")
    out = spec.fit(np.arange(25.0)).forecast(2)
    assert out.shape == (2, 1)
    assert torch.equal(before, torch.cuda.get_rng_state())
