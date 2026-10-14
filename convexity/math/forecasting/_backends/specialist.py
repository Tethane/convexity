"""NeuralForecast data boundaries and global sequence fitting."""

from copy import deepcopy
import random

import numpy as np

from .. import _validation as v
from ..builtins.specialist import GlobalFittedForecaster
from ..interfaces.result import ForecastResult
from ..interfaces.data import ForecastHorizon


def frames(panel):
    import pandas as pd

    rows, statics, coords, schema = [], [], [], None
    for name, series in panel.series.items():
        if not np.isfinite(series.values).all():
            raise ValueError("specialist panel training requires finite targets")
        cov = series.covariates
        widths = {
            key: 0
            if cov is None or getattr(cov, key) is None
            else getattr(cov, key).shape[-1]
            for key in ("known_future", "observed_past", "static")
        }
        if schema is not None and widths != schema:
            raise ValueError("all panel series must have the same covariate widths")
        schema = widths
        for k in range(series.dimension):
            uid = str(len(coords))
            coords.append((uid, name, k))
            frame = pd.DataFrame(
                {
                    "unique_id": uid,
                    "ds": np.arange(len(series)),
                    "y": series.values[:, k],
                }
            )
            for key, prefix in [("known_future", "known"), ("observed_past", "past")]:
                for j in range(widths[key]):
                    frame[f"{prefix}{j}"] = getattr(cov, key)[:, j]
            rows.append(frame)
            statics.append(
                {
                    "unique_id": uid,
                    **{
                        f"static{j}": getattr(cov, "static")[j]
                        for j in range(widths["static"])
                    },
                }
            )
    assert schema is not None
    return (
        pd.concat(rows, ignore_index=True),
        pd.DataFrame(statics) if schema["static"] else None,
        tuple(coords),
        schema,
    )


def train_panel(spec, panel, previous=None):
    import torch
    from neuralforecast import NeuralForecast, models
    from neuralforecast.losses.pytorch import MAE, MQLoss

    df, static, coords, schema = frames(panel)
    if any(
        len(s) < spec.context_length + spec.max_horizon + spec.validation_size
        for s in panel.series.values()
    ):
        raise ValueError(
            "each panel history must cover context, horizon and validation rows"
        )
    options = dict(spec.model_options)
    defaults = {
        "TFT": dict(hidden_size=32, n_head=4),
        "PatchTST": dict(
            hidden_size=32,
            n_heads=4,
            encoder_layers=2,
            patch_len=min(8, spec.context_length),
            stride=2,
        ),
        "NBEATS": dict(mlp_units=[[32, 32]] * 3),
        "NHITS": dict(mlp_units=[[32, 32]] * 3),
        "RNN": dict(encoder_hidden_size=32, decoder_hidden_size=32, encoder_n_layers=1),
    }[spec.model]
    options = {**defaults, **options}
    if spec.model == "NBEATS" and any(schema.values()):
        raise ValueError("NBEATS has no exogenous inputs; TFT/NHiTS support covariates")
    if spec.model != "NBEATS":
        options.update(
            futr_exog_list=[f"known{i}" for i in range(schema["known_future"])],
            hist_exog_list=[f"past{i}" for i in range(schema["observed_past"])],
            stat_exog_list=[f"static{i}" for i in range(schema["static"])],
        )
    options.update(
        h=spec.max_horizon,
        input_size=spec.context_length,
        max_steps=spec.max_steps,
        random_seed=spec.seed,
        loss=MQLoss(quantiles=list(spec.quantiles)) if spec.quantiles else MAE(),
        alias="forecast",
        accelerator="gpu" if spec.device.startswith("cuda") else "cpu",
        devices=1,
    )
    for key, value in dict(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        val_check_steps=max(1, spec.max_steps),
        windows_batch_size=32,
    ).items():
        options.setdefault(key, value)
    py_state, np_state = random.getstate(), np.random.get_state()
    devices = [torch.cuda.current_device()] if spec.device.startswith("cuda") else []
    try:
        with torch.random.fork_rng(devices=devices):
            engine = (
                NeuralForecast([getattr(models, spec.model)(**options)], freq=1)
                if previous is None
                else deepcopy(previous.engine)
            )
            engine.fit(
                df=df,
                static_df=static,
                val_size=spec.validation_size,
                use_init_models=False,
            )
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
    return GlobalFittedForecaster(engine, panel, spec, schema, coords)


def predict_panel(fitted, horizon, covariates, *, align=True):
    import pandas as pd

    h = horizon if isinstance(horizon, ForecastHorizon) else ForecastHorizon(horizon)
    spec = fitted.specification
    if h.maximum > spec.max_horizon:
        raise ValueError("panel forecast exceeds trained horizon")
    unknown = set(covariates) - set(fitted.panel.series)
    if unknown:
        raise ValueError("future covariates contain unknown panel identifiers")
    futures = []
    for uid, name, k in fitted.coordinates:
        history = fitted.panel.series[name]
        row = pd.DataFrame(
            {
                "unique_id": uid,
                "ds": np.arange(len(history), len(history) + spec.max_horizon),
            }
        )
        known = covariates.get(name)
        if fitted.schema["known_future"]:
            known = v.matrix(known, "panel future covariates")
            if known.shape != (spec.max_horizon, fitted.schema["known_future"]):
                raise ValueError(
                    "panel future covariates must cover the full trained horizon"
                )
            for j in range(known.shape[1]):
                row[f"known{j}"] = known[:, j]
        elif known is not None:
            raise ValueError("panel was trained without future covariates")
        futures.append(row)
    df, static, _, _ = frames(fitted.panel)
    raw = fitted.engine.predict(
        df=df, static_df=static, futr_df=pd.concat(futures, ignore_index=True)
    )
    out = {}
    for name, history in fitted.panel.series.items():
        arrays = []
        for uid, series_name, k in fitted.coordinates:
            if series_name == name:
                block = raw[raw.unique_id == uid].sort_values("ds")
                cols = [col for col in block.columns if col.startswith("forecast")]
                arrays.append(block[cols].to_numpy())
        values = np.stack(arrays, axis=1)
        if spec.quantiles:
            # MQLoss names/sorts the requested quantile outputs. Select by suffix
            # rather than assuming provider column order.
            levels = list(spec.quantiles)
            columns = [c for c in raw.columns if c.startswith("forecast")]
            expected = fitted.engine.models[0].loss.output_names
            indices = [columns.index("forecast" + suffix) for suffix in expected]
            q = np.sort(values[..., indices], axis=-1)
            # Monotone rearrangement of separately fitted marginal quantiles.
            # This changes crossing outputs, not the provider's learned weights.
            point = q[..., np.argmin(abs(np.asarray(levels) - 0.5))]
            result = ForecastResult(
                point,
                quantile_levels=levels,
                quantile_values=q,
                metadata={"quantile_postprocessing": "monotone rearrangement"},
            )
        else:
            result = ForecastResult(values[..., 0])
        result = result.select(h.steps - 1, horizon=h.align(history) if align else h)
        from dataclasses import replace

        out[name] = replace(
            result,
            cutoff=history.cutoff,
            model_identity="neuralforecast." + spec.model,
            metadata={
                **result.metadata,
                "series_id": name,
                "global_series_count": len(fitted.panel.series),
                "artifact_identity": fitted.artifact_identity,
                "training_digest": fitted.training_digest,
                "target_names": history.names,
            },
        )
    return out
