import numpy as np
import pytest
from numpy.testing import assert_allclose

from convexity.math import forecasting as f


def test_state_space_joint_covariance_and_incremental_updates():
    rng = np.random.default_rng(4)
    y = np.zeros(200)
    for i in range(1, len(y)):
        y[i] = 0.7 * y[i - 1] + rng.normal()
    fit = f.ARForecaster(trend="n").fit(y)
    out = fit.forecast(4)
    phi, sigma2 = getattr(fit.adapter, "model").params
    assert_allclose(out.covariance()[0], sigma2 * phi ** np.arange(4), atol=1e-8)
    before = fit.predict(2)
    updated = fit.update([1.0, 2.0], mode="incremental")
    assert updated.nobs == 202
    assert_allclose(
        getattr(updated.adapter, "model").params, getattr(fit.adapter, "model").params
    )
    assert_allclose(fit.predict(2), before)
    assert_allclose(updated.predict(3)[:, 0], 2 * phi ** np.arange(1, 4), atol=1e-8)


@pytest.mark.parametrize("factory", [f.ETSForecaster, f.HoltForecaster])
def test_smoothing_has_real_innovation_paths(factory):
    y = np.arange(60.0) * 0.1 + np.random.default_rng(3).normal(size=60) * 0.2
    model = factory(samples=200).fit(y)
    a = model.forecast(4, rng=np.random.default_rng(8))
    b = model.forecast(4, rng=np.random.default_rng(8))
    assert a.paths.shape == (200, 4, 1)
    assert_allclose(a.paths, b.paths)
    assert np.all(a.variance() > 0)


def test_parameter_uncertainty_bootstrap_conditions_on_actual_cutoff():
    y = np.random.default_rng(10).normal(size=100)
    model = f.BootstrapForecaster(
        f.ARForecaster(trend="n"), replications=4, samples=200
    ).fit(y)
    assert np.asarray(model.parameters["parameter_draws"], dtype=float).shape == (4, 2)
    result = model.forecast(3, rng=np.random.default_rng(1))
    assert result.paths is not None
    assert result.paths.shape == (200, 3, 1)
    assert (
        np.std(np.asarray(model.parameters["parameter_draws"], dtype=float)[:, 0]) > 0
    )


def test_elapsed_time_and_business_calendar():
    rng = np.random.default_rng(7)
    times = np.cumsum(rng.uniform(0.2, 2, size=400))
    y = np.zeros(400)
    for i, dt in enumerate(np.diff(times), start=1):
        y[i] = (
            2
            + np.exp(-0.6 * dt) * (y[i - 1] - 2)
            + np.sqrt(0.4**2 * (1 - np.exp(-1.2 * dt)) / 1.2) * rng.normal()
        )
    fitted = f.OUForecaster().fit(f.TimeSeries(y, times))
    parameters = np.asarray(fitted.parameters["mu_theta_sigma"], dtype=float)[0]
    assert_allclose(parameters, [2, 0.6, 0.4], atol=0.15)
    h = f.ForecastHorizon([1, 2], [times[-1] + 0.1, times[-1] + 3])
    out = fitted.forecast(h)
    mu, theta, sigma = parameters
    assert_allclose(
        out.mean()[:, 0], mu + np.exp(-theta * np.array([0.1, 3])) * (y[-1] - mu)
    )
    assert out.variance()[1, 0] > out.variance()[0, 0]
    cal = f.calendar_horizon("2026-10-09", 3, holidays=["2026-10-12"])
    assert cal.timestamps is not None
    assert str(cal.timestamps[0])[:10] == "2026-10-13"
    monthly = f.calendar_horizon("2026-01-31", 2, frequency="ME")
    assert monthly.timestamps is not None
    assert str(monthly.timestamps[0])[:10] == "2026-02-28"


@pytest.fixture
def torch_threads():
    torch = pytest.importorskip("torch", exc_type=ModuleNotFoundError)
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield torch
    torch.set_num_threads(n)


@pytest.mark.parametrize("head", ["student_t", "lognormal", "mixture", "categorical"])
def test_additional_probability_heads(head, torch_threads):
    y = np.sin(np.arange(50.0) / 4)
    if head == "lognormal":
        y = np.exp(y)
    if head == "categorical":
        y = (y > 0).astype(float)
    m = f.TorchForecaster(head=head, context_length=4, max_horizon=2, epochs=3).fit(y)
    result = m.forecast(2)
    assert np.all(result.variance() > 0)
    assert np.all(result.quantile(0.1) <= result.quantile(0.9))
    draws = result.sample(100, rng=np.random.default_rng(1))
    if head == "categorical":
        assert np.isin(draws, [0, 1]).all()
    if head == "lognormal":
        assert np.all(draws > 0)
    assert result.select([1]).quantile(0.5).shape == (1, 1)
    old = m.predict(2)
    values = np.exp([0.1, 0.2]) if head == "lognormal" else [0, 1]
    updated = m.update(values, mode="warm_start")
    assert updated.nobs == 52
    assert_allclose(m.predict(2), old)


@pytest.mark.parametrize(
    "factory",
    [f.FlowForecaster, f.VAEForecaster, f.AutoregressiveForecaster, f.ScoreForecaster],
)
def test_new_generative_models(factory, torch_threads):
    y = np.sin(np.arange(45.0) / 5)
    m = factory(
        context_length=4, max_horizon=2, epochs=4, hidden_size=8, samples=30
    ).fit(y)
    a = m.forecast(2, rng=np.random.default_rng(1))
    b = m.forecast(2, rng=np.random.default_rng(1))
    assert_allclose(a.paths, b.paths)
    assert np.all(a.variance() > 0)
    if factory == f.FlowForecaster:
        torch = torch_threads
        with torch.no_grad():
            c = torch.zeros((5, m.adapter.module.blocks[0][0].in_features - 2))
            values = torch.randn(5, 2)
            z, forward = m.adapter.module.flow(values, c)
            restored, inverse = m.adapter.module.flow(z, c, inverse=True)
            assert_allclose(restored.numpy(), values.numpy(), atol=1e-6)
            assert_allclose((forward + inverse).numpy(), 0, atol=1e-6)


@pytest.mark.parametrize(
    "factory",
    [
        f.TFTForecaster,
        f.PatchTSTForecaster,
        f.NBEATSForecaster,
        f.NHiTSForecaster,
        f.EncoderDecoderForecaster,
    ],
)
def test_dedicated_architectures(factory, torch_threads):
    pytest.importorskip("neuralforecast")
    y = np.sin(np.arange(40.0) / 5)
    m = factory(context_length=8, max_horizon=2, max_steps=2).fit(y)
    assert m.forecast(2).shape == (2, 1)


def test_shared_panel_training_and_warm_start(torch_threads, tmp_path):
    pytest.importorskip("neuralforecast")
    a = np.sin(np.arange(40.0) / 5)
    b = np.cos(np.arange(50.0) / 5)
    panel = f.Panel({"a": a, "b": b})
    m = f.NHiTSForecaster(context_length=8, max_horizon=2, max_steps=2).fit_panel(panel)
    result = m.forecast(2)
    assert result["a"].cutoff == 39 and result["b"].cutoff == 49
    updated = m.update(
        f.Panel(
            {"a": f.TimeSeries([1, 2], [40, 41]), "b": f.TimeSeries([1, 2], [50, 51])}
        ),
        mode="warm_start",
    )
    assert updated.forecast(2)["a"].cutoff == 41
    assert len(m.panel.series["a"]) == 40
    assert updated.training_digest != m.training_digest
    path = tmp_path / "panel.pkl"
    m.save(path)
    restored = f.load_panel_forecaster(path)
    assert restored.artifact_identity == m.artifact_identity
    assert_allclose(restored.forecast(2)["a"].point, result["a"].point)


def test_specialist_quantiles_covariates_and_series_boundaries(torch_threads):
    pytest.importorskip("neuralforecast")
    from convexity.math.forecasting._backends.specialist import frames

    y = np.sin(np.arange(40.0) / 5)
    cov = f.Covariates(known_future=np.arange(40.0), observed_past=y**2, static=[2.0])
    panel = f.Panel(
        {
            "a": f.TimeSeries(y, covariates=cov),
            "b": f.TimeSeries(y + 100, covariates=cov),
        }
    )
    df, _, coords, _ = frames(panel)
    assert df.groupby("unique_id").size().tolist() == [40, 40]
    assert len(coords) == 2
    m = f.NHiTSForecaster(
        context_length=8, max_horizon=2, max_steps=2, quantiles=(0.1, 0.5, 0.9)
    ).fit_panel(panel)
    out = m.forecast(2, covariates={"a": [40.0, 41.0], "b": [40.0, 41.0]})
    assert np.all(out["a"].quantile(0.1) <= out["a"].quantile(0.9))
    assert out["a"].metadata["quantile_postprocessing"] == "monotone rearrangement"
    with pytest.raises(ValueError, match="unknown panel"):
        m.forecast(2, covariates={"wrong": [40.0, 41.0]})  # pyright: ignore[reportArgumentType]  # Exercise runtime rejection.


def test_scalar_laws_cannot_conflict_with_other_probability_representations():
    from convexity.math import distributions as d

    with pytest.raises(ValueError, match="one probability"):
        f.ForecastResult([[0]], paths=np.zeros((2, 1, 1)), marginal_laws=[d.normal()])
    with pytest.raises(ValueError, match="moments"):
        f.ForecastResult([[0]], mean_values=[[1]], marginal_laws=[d.normal()])


def test_head_likelihood_matches_decoded_law(torch_threads):
    from convexity.math.forecasting._backends import heads

    torch = torch_threads
    for head in ("student_t", "mixture", "categorical"):
        spec = f.TorchForecaster(head=head)
        raw = torch.zeros((1, 1, heads.channels(spec)), dtype=torch.float64)
        target = torch.tensor([[0.0]], dtype=torch.float64)
        law = heads.decode(spec, raw, np.array([0.0]), np.array([1.0]))
        assert_allclose(
            float(heads.loss(spec, raw, target)[0, 0]),
            -float(law.marginal(0).log_density(0.0)),
            atol=1e-8,
        )


def test_specialist_accepts_explicit_irregular_forecast_labels(torch_threads):
    pytest.importorskip("neuralforecast")
    times = np.arange(40.0) ** 1.2
    fitted = f.NHiTSForecaster(context_length=8, max_horizon=2, max_steps=2).fit(
        f.TimeSeries(np.sin(np.arange(40.0) / 5), times)
    )
    horizon = f.ForecastHorizon(2, [times[-1] + 1, times[-1] + 3])
    result = fitted.forecast(horizon)
    assert isinstance(result, f.ForecastResult)
    assert result.horizon is not None
    assert result.horizon is not None
    np.testing.assert_array_equal(result.horizon.timestamps, horizon.timestamps)


def test_parameter_bootstrap_preserves_autoregressive_dynamics():
    rng = np.random.default_rng(73)
    y = np.zeros(300)
    for i in range(1, len(y)):
        y[i] = 0.8 * y[i - 1] + rng.normal(scale=0.3)
    fitted = f.BootstrapForecaster(
        f.ARForecaster(trend="n"), replications=12, samples=50, seed=19
    ).fit(y)
    params = np.asarray(fitted.parameters["parameter_draws"], dtype=float)
    assert abs(params[:, 0].mean() - 0.8) < 0.1
    for member in getattr(fitted.adapter, "members"):
        assert_allclose(np.asarray(member.model.model.endog).ravel(), y)
