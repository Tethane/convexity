"""PyTorch sequence windows, training, validation, and forecast decoding."""

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
from torch import nn

from ..builtins.neural import SequenceSchema
from ..interfaces.base import Diagnostics, FittedForecaster
from ..interfaces.data import TimeSeries, prepare
from . import heads
from ..interfaces.result import ForecastResult


class SequenceNetwork(nn.Module):
    def __init__(self, schema, architecture, hidden):
        super().__init__()
        self.schema, self.architecture = schema, architecture
        width = schema.targets * 2 + schema.known + schema.observed
        if architecture == "mlp":
            self.encoder = nn.Sequential(
                nn.Flatten(1),
                nn.Linear(width * schema.context_length, hidden),
                nn.Tanh(),
            )
        elif architecture == "cnn":
            self.encoder = nn.Sequential(
                nn.Conv1d(width, hidden, 3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden, 3, padding=1),
                nn.ReLU(),
            )
        elif architecture in ("rnn", "gru", "lstm"):
            self.encoder = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[
                architecture
            ](width, hidden, batch_first=True)
        else:
            self.projection = nn.Linear(width, hidden)
            self.position = nn.Parameter(torch.zeros(1, schema.context_length, hidden))
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    hidden,
                    nhead=1,
                    dim_feedforward=hidden * 2,
                    dropout=0.0,
                    batch_first=True,
                ),
                num_layers=1,
                enable_nested_tensor=False,
            )
        self.output = nn.Linear(
            hidden + schema.horizon * schema.known + schema.static,
            schema.horizon * schema.targets * schema.channels,
        )

    def forward(self, batch):
        x = torch.cat(
            (
                batch["history"],
                batch["mask"],
                batch["past_known"],
                batch["past_observed"],
            ),
            dim=-1,
        )
        if self.architecture == "cnn":
            hidden = self.encoder(x.transpose(1, 2))[:, :, -1]
        elif self.architecture in ("rnn", "gru", "lstm"):
            hidden = self.encoder(x)[0][:, -1]
        elif self.architecture == "transformer":
            hidden = self.encoder(self.projection(x) + self.position)[:, -1]
        else:
            hidden = self.encoder(x)
        hidden = torch.cat(
            (hidden, batch["future_known"].flatten(1), batch["static"]), dim=-1
        )
        return self.output(hidden).reshape(
            -1, self.schema.horizon, self.schema.targets, self.schema.channels
        )


def make_batch(history, origins, schema, location, scale, future=None):
    y = (history.values - location) / scale
    cov = history.covariates
    known = (
        np.empty((len(history), 0))
        if cov is None or cov.known_future is None
        else cov.known_future
    )
    observed = (
        np.empty((len(history), 0))
        if cov is None or cov.observed_past is None
        else cov.observed_past
    )
    static = np.empty(0) if cov is None or cov.static is None else cov.static
    blocks = np.array([y[o - schema.context_length : o] for o in origins])
    return dict(
        history=np.nan_to_num(blocks),
        mask=np.isfinite(blocks).astype(float),
        past_known=np.array([known[o - schema.context_length : o] for o in origins]),
        past_observed=np.array(
            [observed[o - schema.context_length : o] for o in origins]
        ),
        future_known=(
            np.array([known[o : o + schema.horizon] for o in origins])
            if future is None
            else np.asarray(future)[None]
        ),
        static=np.repeat(static[None], len(origins), axis=0),
    )


def tensors(batch, device):
    return {
        name: torch.tensor(np.array(value), dtype=torch.float32, device=device)
        for name, value in batch.items()
    }


def checked_output(raw, batch_size, schema):
    if not isinstance(raw, torch.Tensor) or tuple(raw.shape) != (
        batch_size,
        schema.horizon,
        schema.targets,
        schema.channels,
    ):
        raise ValueError(
            "neural output must have shape (batch,horizon,targets,channels)"
        )
    if not bool(torch.isfinite(raw).all()):
        raise FloatingPointError("neural output contains nonfinite values")
    return raw


def ordered_quantiles(raw):
    return torch.cat(
        (
            raw[..., :1],
            raw[..., :1] + torch.cumsum(nn.functional.softplus(raw[..., 1:]), dim=-1),
        ),
        dim=-1,
    )


def objective(spec, raw, target, mask):
    if spec.head == "custom":
        loss = spec.loss(raw, target, mask)
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            raise ValueError("custom loss must return a scalar torch tensor")
        return loss
    if spec.head in ("student_t", "lognormal", "mixture", "categorical"):
        loss = heads.loss(spec, raw, target)
    elif spec.head == "point":
        loss = (raw[..., 0] - target) ** 2
    elif spec.head == "gaussian":
        sigma = nn.functional.softplus(raw[..., 1]) + 1e-6
        loss = (
            torch.log(sigma)
            + 0.5 * ((target - raw[..., 0]) / sigma) ** 2
            + 0.5 * np.log(2 * np.pi)
        )
    else:
        error = target[..., None] - ordered_quantiles(raw)
        levels = torch.tensor(spec.quantiles, device=raw.device, dtype=raw.dtype)
        loss = torch.maximum(levels * error, (levels - 1) * error).mean(dim=-1)
    return (loss * mask).sum() / mask.sum().clamp_min(1)


@dataclass(frozen=True)
class TorchAdapter:
    module: Any
    history: Any
    schema: Any
    spec: Any
    location: Any
    scale: Any

    def update(self, history, *, mode):
        if mode == "incremental":
            # New observations move the context window, without learning from
            # the evaluation period or changing the training normalization.
            prepared = prepare_neural_history(self.spec, history)
            adapter = replace(self, history=prepared)
            return FittedForecaster(
                adapter,
                history,
                self.spec,
                Diagnostics(
                    "torch.context",
                    message="Context updated; weights and scaling unchanged",
                ),
                {
                    "schema": self.schema,
                    "location": self.location,
                    "scale": self.scale,
                    "head": self.spec.head,
                    "device": self.spec.device,
                },
                "torch." + self.spec.architecture + "." + self.spec.head,
                self.spec.seed,
            )
        if mode != "warm_start":
            raise ValueError("neural updates use mode=incremental or mode=warm_start")
        return train(self.spec, history, None, previous=self)

    def forecast(self, steps, covariates, *, rng=None):
        if steps > self.schema.horizon:
            raise ValueError("requested horizon exceeds the trained neural horizon")
        if self.schema.known and steps != self.schema.horizon:
            raise ValueError(
                "neural models with future covariates require the full trained horizon"
            )
        future = (
            np.empty((self.schema.horizon, 0)) if covariates is None else covariates
        )
        batch = make_batch(
            self.history,
            [len(self.history)],
            self.schema,
            self.location,
            self.scale,
            future,
        )
        self.module.eval()
        with torch.no_grad():
            raw = checked_output(
                self.module(tensors(batch, self.spec.device)), 1, self.schema
            )
            if self.spec.head == "custom":
                result = self.spec.decoder(
                    raw, self.location.copy(), self.scale.copy(), rng=rng
                )
                if not isinstance(result, ForecastResult) or result.shape != (
                    self.schema.horizon,
                    self.schema.targets,
                ):
                    raise ValueError(
                        "custom decoder must return a ForecastResult in original target units"
                    )
                return result.select(np.arange(steps))
            if self.spec.head in ("student_t", "lognormal", "mixture", "categorical"):
                return heads.decode(
                    self.spec, raw[0, :steps], self.location, self.scale
                )
            point = raw[0, ..., 0].cpu().numpy() * self.scale + self.location
            if self.spec.head == "point":
                return ForecastResult(point[:steps], mean_values=point[:steps])
            if self.spec.head == "gaussian":
                sigma = (
                    nn.functional.softplus(raw[0, ..., 1]) + 1e-6
                ).cpu().numpy() * self.scale
                return ForecastResult(
                    point[:steps],
                    mean_values=point[:steps],
                    gaussian=True,
                    joint_covariance=np.diag(sigma[:steps].ravel() ** 2),
                    metadata={"uncertainty": "conditionally independent Gaussian head"},
                )
            q = (
                ordered_quantiles(raw)[0].cpu().numpy() * self.scale[None, :, None]
                + self.location[None, :, None]
            )
            levels = np.asarray(self.spec.quantiles)
            median = np.flatnonzero(levels == 0.5)
            point = (
                q[..., median[0]]
                if len(median)
                else q[..., np.argmin(abs(levels - 0.5))]
            )
            return ForecastResult(
                point[:steps], quantile_levels=levels, quantile_values=q[:steps]
            )


def prepare_neural_history(spec, data, covariates=None):
    history = prepare(
        data, covariates, missing="native" if spec.missing == "mask" else spec.missing
    )
    if spec.head == "lognormal":
        if np.any(history.values <= 0):
            raise ValueError("lognormal targets must be strictly positive")
        history = TimeSeries(
            np.log(history.values),
            history.timestamps,
            history.covariates,
            history.names,
        )
    if spec.head == "categorical" and not spec.discretize:
        observed = history.values[np.isfinite(history.values)]
        if not np.isin(observed, spec.category_values).all():
            raise ValueError(
                "categorical targets must be on category_values; use discretize=True for bin representatives"
            )
    return history


def train(spec, data, covariates, previous=None):
    original_history = prepare(
        data, covariates, missing="native" if spec.missing == "mask" else spec.missing
    )
    history = prepare_neural_history(spec, original_history)
    cutoff = len(history) - spec.validation_size
    if cutoff < spec.context_length + spec.max_horizon:
        raise ValueError(
            "insufficient training rows for context, horizon, and validation holdout"
        )
    if spec.validation_size and spec.validation_size < spec.max_horizon:
        raise ValueError(
            "validation_size must cover at least one complete forecast horizon"
        )
    prefix = history.values[:cutoff]
    if np.any(np.all(np.isnan(prefix), axis=0)):
        raise ValueError(
            "every target requires observed training values before validation"
        )
    location = np.nanmean(prefix, axis=0)
    scale = np.nanstd(prefix, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    if spec.head == "categorical":
        location, scale = np.zeros(history.dimension), np.ones(history.dimension)
    if previous is not None:
        location, scale = previous.location.copy(), previous.scale.copy()
    cov = history.covariates
    dims = [
        0 if cov is None or getattr(cov, name) is None else getattr(cov, name).shape[-1]
        for name in ("known_future", "observed_past", "static")
    ]
    channels = heads.channels(spec)
    schema = SequenceSchema(
        spec.context_length,
        spec.max_horizon,
        history.dimension,
        dims[0],
        dims[1],
        dims[2],
        channels,
    )
    origins = np.arange(spec.context_length, cutoff - spec.max_horizon + 1)
    validation = (
        np.arange(cutoff, len(history) - spec.max_horizon + 1)
        if spec.validation_size
        else np.array([], dtype=int)
    )
    target_values = (history.values - location) / scale

    def dataset(rows):
        batch = tensors(make_batch(history, rows, schema, location, scale), spec.device)
        target = np.array([target_values[o : o + schema.horizon] for o in rows])
        mask = torch.tensor(
            np.isfinite(target), device=spec.device, dtype=torch.float32
        )
        target = torch.tensor(
            np.nan_to_num(target), device=spec.device, dtype=torch.float32
        )
        keep = mask.sum(dim=(1, 2)) > 0
        if not bool(keep.any()):
            raise ValueError(
                "no observed targets in the requested training/validation windows"
            )
        return {k: x[keep] for k, x in batch.items()}, target[keep], mask[keep]

    training = dataset(origins)
    valid = dataset(validation) if len(validation) else None
    device = torch.device(spec.device)
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    rng = np.random.default_rng(spec.seed)
    losses, validation_losses = [], []
    best, best_loss, stale = None, np.inf, 0
    with torch.random.fork_rng(devices=devices):
        cpu_rng = torch.Generator(device="cpu").manual_seed(spec.seed)
        torch.set_rng_state(cpu_rng.get_state())
        if devices:
            gpu_rng = torch.Generator(device=device).manual_seed(spec.seed)
            torch.cuda.set_rng_state(gpu_rng.get_state(), device=device)
        module = (
            SequenceNetwork(schema, spec.architecture, spec.hidden_size)
            if spec.module_factory is None
            else spec.module_factory(schema)
        )
        if not isinstance(module, nn.Module):
            raise TypeError("module_factory must return torch.nn.Module")
        if previous is not None:
            module.load_state_dict(previous.module.state_dict())
        module = deepcopy(module).to(device=device, dtype=torch.float32)
        for parameter in module.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.Adam(module.parameters(), lr=spec.learning_rate)
        for epoch in range(spec.epochs):
            module.train()
            order = rng.permutation(len(training[1]))
            total, count = 0.0, 0
            for start in range(0, len(order), spec.batch_size):
                idx = torch.tensor(
                    order[start : start + spec.batch_size], device=device
                )
                batch, target, mask = training
                optimizer.zero_grad()
                raw = checked_output(
                    module({k: x[idx] for k, x in batch.items()}), len(idx), schema
                )
                loss = objective(spec, raw, target[idx], mask[idx])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite neural training loss")
                loss.backward()
                if any(
                    p.grad is not None and not bool(torch.isfinite(p.grad).all())
                    for p in module.parameters()
                ):
                    raise FloatingPointError("nonfinite neural gradient")
                optimizer.step()
                total += float(loss.detach()) * len(idx)
                count += len(idx)
            losses.append(total / count)
            if valid is not None:
                module.eval()
                with torch.no_grad():
                    batch, target, mask = valid
                    raw = checked_output(module(batch), len(target), schema)
                    score = float(objective(spec, raw, target, mask))
                if not np.isfinite(score):
                    raise FloatingPointError("nonfinite validation loss")
                validation_losses.append(score)
                if score < best_loss:
                    best, best_loss, stale = deepcopy(module.state_dict()), score, 0
                else:
                    stale += 1
                if stale >= spec.patience:
                    break
        if best is not None:
            module.load_state_dict(best)
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    diagnostics = Diagnostics(
        "torch.Adam",
        message="Training budget/early stopping; no optimality claim",
        iterations=len(losses),
        loss_history=losses,
        validation_history=validation_losses or None,
    )
    return FittedForecaster(
        TorchAdapter(module, history, schema, spec, location, scale),
        original_history,
        spec,
        diagnostics,
        {
            "schema": schema,
            "location": location,
            "scale": scale,
            "head": spec.head,
            "device": spec.device,
        },
        "torch." + spec.architecture + "." + spec.head,
        spec.seed,
    )
