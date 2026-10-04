"""Executable and static consumer checks for the SDK's public type contracts."""

import inspect
import re
from dataclasses import replace
from typing import assert_type

import numpy as np
import pytest
from numpy.testing import assert_allclose

import convexity.math as sdk
from convexity.math import (
    distributions,
    estimation,
    forecasting,
    numerical,
    pricing,
    regimes,
)
from convexity.math.types import ArrayLike, FloatArray


@pytest.mark.parametrize("family", sdk.__all__)
def test_exported_callables_documented_without_explicit_any(family):
    module = getattr(sdk, family)
    for name in module.__all__:
        value = getattr(module, name)
        if not callable(value) or not getattr(value, "__module__", "").startswith(
            "convexity.math"
        ):
            continue
        assert value.__doc__ and not value.__doc__.startswith(f"{name}("), (
            f"{family}.{name} needs an authored docstring"
        )
        assert inspect.getdoc(value), f"{family}.{name} needs a public contract"
        try:
            signature = inspect.signature(value)
        except ValueError:
            assert inspect.isclass(value) and issubclass(value, BaseException)
            continue
        if not getattr(value, "_is_protocol", False):
            for parameter in signature.parameters.values():
                assert parameter.annotation is not inspect.Signature.empty, (
                    f"{family}.{name}.{parameter.name} needs a type"
                )
                assert not re.search(
                    r"\b(?:Callable|dict|tuple|list|Mapping|Sequence)\b(?!\[)",
                    str(parameter.annotation),
                ), f"{family}.{name}.{parameter.name} needs parameterized types"

            assert signature.return_annotation is not inspect.Signature.empty, (
                f"{family}.{name} needs a return type"
            )
        assert not re.search(r"\bAny\b", str(signature)), (
            f"{family}.{name}: {signature}"
        )


def test_owned_arrays_accept_lists_and_preserve_snapshot_and_replace_semantics():
    raw = [1.0, 2.0]
    estimate = estimation.Estimate(
        raw,
        names=["a", "b"],
        covariance=[[2.0, 0.0], [0.0, 3.0]],
        covariance_kind="normal",
    )
    assert_type(estimate.parameters, FloatArray)
    raw[0] = 100.0
    assert_allclose(estimate.parameters, [1.0, 2.0])
    assert not estimate.parameters.flags.writeable
    updated = replace(estimate, parameters=[3.0, 4.0])
    assert_allclose(updated.parameters, [3.0, 4.0])
    assert_allclose(estimate.parameters, [1.0, 2.0])
    assert updated.names == ("a", "b")
    assert updated.covariance is not None
    assert_allclose(updated.covariance, [[2.0, 0.0], [0.0, 3.0]])

    from datetime import date
    from types import MappingProxyType

    from convexity.math import contracts

    raw_history = [1.0, 3.0]
    panel = forecasting.Panel(MappingProxyType({"series": raw_history}))
    assert_type(panel.series["series"], forecasting.TimeSeries)
    raw_history[-1] = 100.0
    assert_allclose(panel.series["series"].values, [[1.0], [3.0]])
    assert contracts.Exercise("european", ["2026-09-30"]) == contracts.Exercise(
        "european", [date(2026, 9, 30)]
    )


def test_grid_contract_is_an_array_and_known_backend_route_preserves_it():
    grid = numerical.linspace(0.0, 1.0, 5)
    assert_type(grid, FloatArray)
    routed = numerical.scipy_backend().run(
        "linspace", numerical.GridProblem("linear", 0.0, 1.0, 5)
    )
    assert_type(routed, FloatArray)
    assert_allclose(grid, [0.0, 0.25, 0.5, 0.75, 1.0])
    assert_allclose(routed, grid)
    assert not grid.flags.writeable


def test_custom_event_and_pricing_contracts_need_no_provider_objects():
    class Clock:
        def sample(
            self, start: float, end: float, *, rng: np.random.Generator
        ) -> ArrayLike:
            return [end] if end > start else []

    events = regimes.sample_events(Clock(), 0.0, 2.0, rng=np.random.default_rng(1))
    assert_type(events, FloatArray)
    assert_allclose(events, [2.0])

    class Model:
        measure = "Q"

    class Engine:
        def price(self, problem: pricing.PricingProblem) -> pricing.PricingResult:
            return pricing.PricingResult(7.0, engine="user")

    problem = pricing.PricingProblem(object(), Model(), "2026-09-30")
    assert isinstance(problem.model, Model)
    assert pricing.price(problem, Engine()).value == 7.0


def test_explicit_forecast_callbacks_and_sparse_horizon():
    def predict(
        history: forecasting.TimeSeries,
        steps: int,
        future_known: FloatArray | None,
        *,
        rng: np.random.Generator | None,
    ) -> forecasting.ForecastResult:
        return forecasting.ForecastResult(np.repeat(history.values[-1:], steps, axis=0))

    callback: forecasting.ForecastFunction = predict
    result = forecasting.FoundationForecaster(callback).fit([1.0, 3.0]).forecast([1, 4])
    assert_allclose(result.point, [[3.0], [3.0]])
    assert "sample" not in result.capabilities
    assert distributions.normal(np.array(2.0), np.array(3.0)).mean() == 2.0


def test_default_factories_remain_independent():
    first = pricing.PricingResult(1.0)
    second = pricing.PricingResult(2.0)
    assert first.greeks is not second.greeks
    assert dict(first.greeks) == dict(second.greeks) == {}
    revised = replace(first, greeks={"delta": 0.5})
    assert dict(first.greeks) == {}
    assert revised.greeks["delta"] == 0.5


@pytest.mark.parametrize("workflow", ["control", "liquidation"])
def test_solver_cannot_claim_optimal_without_a_primal_solution(monkeypatch, workflow):
    import cvxpy as cp

    from convexity.math import control, execution

    def broken_solve(problem, **kwargs):
        problem._status = "optimal"

    monkeypatch.setattr(cp.Problem, "solve", broken_solve)
    with pytest.raises(RuntimeError, match="no primal solution"):
        if workflow == "control":
            tree = control.ScenarioTree([-1, 0], [1.0, 1.0], [[0.0], [0.0]])
            control.scenario_mpc(tree, [[1.0]], [[1.0]], [[1.0]], [[1.0]], [1.0])
        else:
            execution.liquidate([1.0], [1.0], impact=[[1.0]], covariance=[[1.0]])


def test_fitted_result_types_survive_provider_dispatch():
    from convexity.math import dependence, learning, transforms

    x = [[0.0], [1.0], [2.0], [3.0]]
    fitted = learning.LinearRegression().fit(x, [1.0, 3.0, 5.0, 7.0])
    assert_type(fitted, learning.FittedModel)
    prediction = fitted.predict_mean([[4.0]])
    assert_type(prediction, FloatArray)
    assert_allclose(prediction, [9.0], atol=1e-12)
    covariance = dependence.SampleCovariance().fit(x)
    assert_type(covariance, dependence.CovarianceEstimate)
    assert_allclose(covariance.covariance, [[5.0 / 3.0]])
    scaling = transforms.Standardize().fit(x)
    assert_type(scaling, transforms.FittedScaling)
    assert_allclose(scaling.inverse_transform(scaling.transform(x)), x)


def test_forecast_comparison_accepts_read_only_public_model_mapping():
    from types import MappingProxyType

    models: dict[str, forecasting.Forecaster] = {"drift": forecasting.DriftForecaster()}
    result = forecasting.compare(
        MappingProxyType(models),
        np.arange(12.0),
        initial_window=5,
        horizon=2,
        include_baselines=False,
    )
    assert set(result) == {"drift"}
    assert_allclose(result["drift"].score(), 0.0, atol=1e-12)
    assert_type(result["drift"].score(), float)
    assert_type(result["drift"].score(by_horizon=True), FloatArray)


def test_generated_public_reference_is_current():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [sys.executable, str(root / "scripts/build_math_reference.py"), "--check"],
        cwd=root,
        check=True,
    )
