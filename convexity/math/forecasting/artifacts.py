"""Versioned, pickle-free numerical fitted artifacts (JSON + NumPy arrays).

Version 1 stores public numerical state, never Python object memory layouts.
Optional engines can register explicitly trusted codecs without executing code
named by an untrusted artifact. Unknown versions/types are rejected.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Callable, Sequence, Mapping
from numpy.typing import ArrayLike, DTypeLike
from .._typing import FloatArray, NumericArray, BoolArray, IntArray, DateLike

if TYPE_CHECKING:
    from . import FittedForecaster, ForecastAdapter, TimeSeries

from typing import TYPE_CHECKING
from collections.abc import Callable, Sequence, Mapping
from numpy.typing import ArrayLike, DTypeLike
from .._typing import FloatArray, NumericArray, BoolArray, IntArray, DateLike

if TYPE_CHECKING:
    from . import FittedForecaster
from os import PathLike

from typing import TYPE_CHECKING
from collections.abc import Callable, Sequence, Mapping
from numpy.typing import ArrayLike, DTypeLike
from .._typing import FloatArray, NumericArray, BoolArray, IntArray, DateLike

if TYPE_CHECKING:
    from . import FittedForecaster
from os import PathLike

from dataclasses import fields, replace
from io import BytesIO
import json
import zipfile
from typing import Any

import numpy as np

from .interfaces.base import Diagnostics, FittedForecaster
from .builtins.neural import SequenceSchema
from .interfaces.data import Covariates, TimeSeries

_CODECS = {}


def register_artifact_codec(
    name: str,
    adapter_type: type[ForecastAdapter],
    encode: Callable[[FittedForecaster], Mapping[str, object]],
    decode: Callable[[Mapping[str, object], TimeSeries], FittedForecaster],
) -> None:
    """Register trusted local codec functions; artifact files cannot import code."""
    if not isinstance(name, str) or not name or name in _CODECS:
        raise ValueError("codec name must be nonempty and not already registered")
    if (
        not callable(encode)
        or not callable(decode)
        or not isinstance(adapter_type, type)
    ):
        raise TypeError("codec requires an adapter type and encode/decode callables")
    _CODECS[name] = (adapter_type, encode, decode)


def _spec(spec):
    config = {f.name: getattr(spec, f.name) for f in fields(spec) if f.init}
    return {"class": type(spec).__name__, "config": config}


def _make_spec(payload):
    from .builtins import baselines, classical, neural, generative, elapsed, regression

    classes = [
        getattr(module, name)
        for module in (baselines, classical, neural, generative, elapsed, regression)
        for name in dir(module)
        if name.endswith("Forecaster") and isinstance(getattr(module, name), type)
    ]
    registry = {c.__name__: c for c in classes}
    if payload["class"] not in registry:
        raise ValueError("unsupported portable specification class")
    config = dict(payload["config"])
    if payload["class"] == "RegressionForecaster" and config["estimator"] is not None:
        from sklearn.linear_model import Ridge, LinearRegression, Lasso, ElasticNet

        estimators = {
            cls.__name__: cls for cls in (Ridge, LinearRegression, Lasso, ElasticNet)
        }
        entry = config["estimator"]
        config["estimator"] = estimators[entry["class"]](**entry["config"])
    return registry[payload["class"]](**config)


def _history(history):
    cov = history.covariates
    return {
        "values": history.values,
        "timestamps": history.timestamps,
        "names": history.names,
        "covariates": None
        if cov is None
        else {f.name: getattr(cov, f.name) for f in fields(cov)},
    }


def _restore_history(raw):
    return TimeSeries(
        raw["values"],
        raw["timestamps"],
        None if raw["covariates"] is None else Covariates(**raw["covariates"]),
        tuple(raw["names"]),
    )


def save_portable(fitted: FittedForecaster, path: str | PathLike[str]) -> None:
    """Write portable numerical state for built-in engines or registered codecs.

    Supported built-ins: baselines, OU, linear lag regression, state-space
    classical models, all native neural heads/architectures, DDPM, flow, VAE,
    autoregressive and score models. No optimization runs when they are loaded.
    """
    # Engine-name dispatch reads provider-specific fitted state only in this codec.
    adapter: Any = fitted.adapter
    name = type(adapter).__name__
    state = {}
    spec: Any = None
    if name in (
        "_BaselineAdapter",
        "_OUAdapter",
        "ClassicalAdapter",
        "_RegressionAdapter",
        "TorchAdapter",
        "DiffusionAdapter",
        "ConditionalGeneratorAdapter",
    ):
        spec = _spec(fitted.specification)
    if name == "_BaselineAdapter":
        kind = "baseline"
    elif name == "_OUAdapter":
        kind, state = "ou", {"parameters": adapter.parameters}
    elif name == "ClassicalAdapter" and hasattr(adapter.model, "filter_results"):
        kind, state = "statespace", {"params": np.asarray(adapter.model.params)}
    elif name == "_RegressionAdapter":
        kind = "linear-regression"
        estimator = spec["config"]["estimator"]
        if estimator is not None:
            from sklearn.linear_model import Ridge, LinearRegression, Lasso, ElasticNet

            if type(estimator) not in (Ridge, LinearRegression, Lasso, ElasticNet):
                raise ValueError(
                    "portable regression specification requires Ridge, LinearRegression, Lasso or ElasticNet"
                )
            spec["config"]["estimator"] = {
                "class": type(estimator).__name__,
                "config": estimator.get_params(),
            }
        models = []
        for model in adapter.models:
            if not hasattr(model, "coef_") or not hasattr(model, "intercept_"):
                raise ValueError(
                    "portable regression currently requires fitted affine sklearn regressors"
                )
            models.append(
                {
                    "coefficients": np.asarray(model.coef_),
                    "intercept": np.asarray(model.intercept_),
                }
            )
        state = {"models": models}
    elif name in ("TorchAdapter", "DiffusionAdapter", "ConditionalGeneratorAdapter"):
        if (
            getattr(adapter.spec, "module_factory", None) is not None
            or getattr(adapter.spec, "head", "") == "custom"
        ):
            raise ValueError(
                "custom neural architectures/heads require a registered portable codec"
            )
        kind = {
            "TorchAdapter": "torch",
            "DiffusionAdapter": "ddpm",
            "ConditionalGeneratorAdapter": "generative",
        }[name]
        state = {
            "weights": {
                k: value.detach().cpu().numpy()
                for k, value in adapter.module.state_dict().items()
            },
            "location": adapter.location,
            "scale": adapter.scale,
        }
        if kind == "ddpm":
            state["prediction_type"] = adapter.prediction_type
            if adapter.prediction_type == "v_prediction":
                # Old readers must reject this engine, not interpret velocity
                # weights as epsilon. The container schema itself is unchanged.
                kind = "ddpm-velocity"
        if kind == "torch":
            state["schema"] = {
                f.name: getattr(adapter.schema, f.name) for f in fields(adapter.schema)
            }
            state["adapter_history"] = _history(adapter.history)
    else:
        for key, (cls, encode, _) in _CODECS.items():
            if isinstance(adapter, cls):
                kind, state = "custom:" + key, encode(fitted)
                spec = None
                break
        else:
            raise ValueError(
                "this engine needs its native checkpoint or an explicitly registered portable codec"
            )
    payload = {
        "format": "convexity.forecast.numerical",
        "version": 1,
        "kind": kind,
        "state": state,
        "specification": spec,
        "history": _history(fitted.history),
        "model_identity": fitted.model_identity,
        "artifact_identity": fitted.artifact_identity,
        "random_state": fitted.random_state,
        "dependencies": fitted.dependency_versions,
        "training_digest": fitted.training_digest,
        "parameters": fitted.parameters,
        "diagnostics": {
            f.name: getattr(fitted.diagnostics, f.name) for f in fields(Diagnostics)
        },
    }
    arrays = {}

    def encode(value):
        if isinstance(value, SequenceSchema):
            return {
                "sequence_schema": {
                    f.name: getattr(value, f.name) for f in fields(value)
                }
            }
        if isinstance(value, np.ndarray):
            name = f"array{len(arrays)}"
            arrays[name] = value
            return {"array": name}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (list, tuple)):
            return [encode(x) for x in value]
        if isinstance(value, dict):
            return {key: encode(val) for key, val in value.items()}
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(
            f"portable artifacts cannot store {type(value).__name__}; use a registered codec"
        )

    manifest = json.dumps(encode(payload), allow_nan=False)
    buffer = BytesIO()
    np.savez(buffer, **arrays)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", manifest)
        archive.writestr("arrays.npz", buffer.getvalue())


def load_portable(
    path: str | PathLike[str], *, device: str = "cpu"
) -> FittedForecaster:
    """Reconstruct numerical state with current compatible implementations.

    The schema is independent of Python pickle and backend object layouts.
    Mathematical parameter/architecture definitions must remain compatible;
    arbitrary future third-party API changes are not silently accepted.
    """
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        with np.load(BytesIO(archive.read("arrays.npz")), allow_pickle=False) as arrays:

            def decode(value) -> Any:
                if isinstance(value, dict):
                    if set(value) == {"sequence_schema"}:
                        return SequenceSchema(**value["sequence_schema"])
                    if set(value) == {"array"}:
                        return np.array(arrays[value["array"]])
                    return {key: decode(val) for key, val in value.items()}
                if isinstance(value, list):
                    return [decode(x) for x in value]
                return value

            payload = decode(manifest)
    if (
        payload.get("format") != "convexity.forecast.numerical"
        or payload.get("version") != 1
    ):
        raise ValueError("unsupported portable artifact format/version")
    history, state, kind = (
        _restore_history(payload["history"]),
        payload["state"],
        payload["kind"],
    )
    if kind.startswith("custom:"):
        name = kind.removeprefix("custom:")
        if name not in _CODECS:
            raise ValueError("required portable codec is not registered")
        fitted = _CODECS[name][2](state, history)
    else:
        spec = _make_spec(payload["specification"])
        if kind == "baseline":
            fitted = spec.fit(history)
        elif kind == "ou":
            from .builtins.elapsed import _OUAdapter

            fitted = FittedForecaster(
                _OUAdapter(history, state["parameters"], spec), history, spec
            )
        elif kind == "statespace":
            from ._backends.statsmodels import ClassicalAdapter
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            from statsmodels.tsa.statespace.varmax import VARMAX
            from statsmodels.tsa.statespace.structural import UnobservedComponents
            from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

            factory = {
                "arima": ARIMA,
                "sarimax": SARIMAX,
                "varmax": VARMAX,
                "structural": UnobservedComponents,
                "dynamic_factor": DynamicFactor,
            }[spec.model]
            cov = history.covariates
            x = None if cov is None else cov.known_future
            y = history.values[:, 0] if history.dimension == 1 else history.values
            model = factory(y, exog=x, **spec.model_options).filter(state["params"])
            fitted = FittedForecaster(
                ClassicalAdapter(model, spec.model, history.values, spec), history, spec
            )
        elif kind == "linear-regression":
            from .builtins.regression import _RegressionAdapter

            models = tuple(
                _AffinePredictor(x["coefficients"], x["intercept"])
                for x in state["models"]
            )
            fitted = FittedForecaster(
                _RegressionAdapter(models, history, spec), history, spec
            )
        else:
            import torch

            spec = replace(spec, device=device)
            with torch.random.fork_rng(devices=[]):
                if kind == "torch":
                    from ._backends.torch import SequenceNetwork, TorchAdapter

                    schema = SequenceSchema(**state["schema"])
                    module = SequenceNetwork(
                        schema, spec.architecture, spec.hidden_size
                    )
                    adapter = TorchAdapter(
                        module,
                        _restore_history(state["adapter_history"]),
                        schema,
                        spec,
                        state["location"],
                        state["scale"],
                    )
                elif kind in ("ddpm", "ddpm-velocity", "generative"):
                    from ._backends.diffusion import (
                        condition,
                        Denoiser,
                        DiffusionAdapter,
                    )
                    from ._backends.generative import (
                        ConditionalGenerator,
                        ConditionalGeneratorAdapter,
                    )

                    context = condition(
                        history,
                        len(history) - spec.max_horizon,
                        spec,
                        state["location"],
                        state["scale"],
                    )
                    width = spec.max_horizon * history.dimension
                    if kind in ("ddpm", "ddpm-velocity"):
                        module = Denoiser(len(context), width, spec.hidden_size)
                        prediction_type = state.get("prediction_type", "epsilon")
                        if prediction_type not in ("epsilon", "v_prediction"):
                            raise ValueError("unsupported DDPM prediction_type")
                        if (
                            kind == "ddpm-velocity"
                            and prediction_type != "v_prediction"
                        ):
                            raise ValueError(
                                "DDPM velocity artifact must declare v_prediction"
                            )
                        adapter = DiffusionAdapter(
                            module,
                            history,
                            spec,
                            state["location"],
                            state["scale"],
                            prediction_type,
                        )
                    else:
                        module = ConditionalGenerator(spec, len(context), width)
                        adapter = ConditionalGeneratorAdapter(
                            module, history, spec, state["location"], state["scale"]
                        )
                else:
                    raise ValueError("unsupported numerical engine kind")
            module.load_state_dict(
                {key: torch.as_tensor(value) for key, value in state["weights"].items()}
            )
            module.to(device=device).eval()
            fitted = FittedForecaster(adapter, history, spec)
    if fitted.training_digest != payload["training_digest"]:
        raise ValueError("portable artifact training-data digest mismatch")
    return replace(
        fitted,
        model_identity=payload["model_identity"],
        artifact_identity=payload["artifact_identity"],
        random_state=payload["random_state"],
        dependency_versions=payload["dependencies"],
        parameters=payload["parameters"],
        diagnostics=Diagnostics(**payload["diagnostics"]),
    )


class _AffinePredictor:
    def __init__(self, coefficients, intercept):
        self.coef_, self.intercept_ = coefficients, intercept

    def predict(self, x):
        return np.asarray(x) @ self.coef_.T + self.intercept_
