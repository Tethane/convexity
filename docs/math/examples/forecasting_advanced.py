"""Expanded family-8 examples, using generated data and small training budgets.

Requires convexity[forecasting-advanced]. For pretrained weights, run the separate
forecasting_pretrained.py integration script. No forecast quality is asserted.
"""

from pathlib import Path

import numpy as np

from convexity.math import forecasting as f


def main():
    import torch

    torch.set_num_threads(2)
    y = np.sin(np.arange(80.0) / 6) + np.random.default_rng(3).normal(0, 0.1, 80)
    for factory in (
        f.FlowForecaster,
        f.VAEForecaster,
        f.AutoregressiveForecaster,
        f.ScoreForecaster,
    ):
        fitted = factory(context_length=8, max_horizon=3, epochs=3, samples=50).fit(y)
        result = fitted.forecast(3, rng=np.random.default_rng(7))
        assert result.paths is not None
        print(fitted.model_identity, result.paths.shape, result.interval(0.8).shape)

    neural = f.TorchForecaster(
        head="student_t", context_length=8, max_horizon=3, epochs=3
    ).fit(y)
    print("Student-t medians:", neural.forecast(3).quantile(0.5).ravel())
    updated = neural.update([0.1, 0.2], mode="warm_start")
    assert updated.nobs == 82 and neural.nobs == 80
    path = Path(".cache/forecasting/student-t.cvf")
    path.parent.mkdir(parents=True, exist_ok=True)
    f.save_portable(updated, path)
    np.testing.assert_allclose(f.load_portable(path).predict(3), updated.predict(3))

    statistical = f.ARForecaster().fit(y)
    print(
        "Incremental cutoff:", statistical.update([0.1, 0.2], mode="incremental").cutoff
    )
    uncertain = f.BootstrapForecaster(
        f.ARForecaster(), replications=4, samples=100
    ).fit(y)
    print(
        "Bootstrap intervals:",
        uncertain.forecast(3, rng=np.random.default_rng(7)).interval(0.8)[:, 0],
    )

    dates = np.datetime64("2026-06-01") + np.arange(80).astype("timedelta64[D]")
    history = f.TimeSeries(y, dates)
    horizon = f.calendar_horizon(history.cutoff, 3, holidays=["2026-09-07"])
    elapsed_forecast = f.OUForecaster().fit(history).forecast(horizon)
    assert elapsed_forecast.horizon is not None
    print("Calendar OU times:", elapsed_forecast.horizon.timestamps)

    panel = f.Panel({"sine": y, "offset": y + 2})
    model = f.NHiTSForecaster(context_length=8, max_horizon=3, max_steps=3)
    global_fit = model.fit_panel(panel)
    print(
        "Global forecasts:",
        {name: value.point.ravel() for name, value in global_fit.forecast(3).items()},
    )
    print("Advanced examples complete; portable checkpoint:", path)


if __name__ == "__main__":
    main()
