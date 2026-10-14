"""Conditional DDPM velocity prediction and reverse-time path generation."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .. import _validation as v
from ..interfaces.base import Diagnostics, FittedForecaster
from ..interfaces.data import prepare
from ..interfaces.result import ForecastResult


class Denoiser(nn.Module):
    def __init__(self, condition_width, path_width, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(condition_width + path_width + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, path_width),
        )

    def forward(self, noisy, time, condition):
        return self.net(torch.cat((noisy, time, condition), dim=-1))


def schedule(steps):
    # Cosine cumulative signal schedule reaches near-pure noise even for a short
    # diffusion chain; clipping prevents division by zero in the reverse mean.
    t = np.linspace(0, 1, steps + 1)
    cumulative = np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
    cumulative /= cumulative[0]
    beta = np.minimum(1 - cumulative[1:] / cumulative[:-1], 0.999)
    return beta, np.cumprod(1 - beta)


def condition(history, origin, spec, location, scale, future=None):
    y = (history.values[origin - spec.context_length : origin] - location) / scale
    parts = [y.ravel()]
    cov = history.covariates
    if cov is not None:
        for name in ("known_future", "observed_past"):
            x = getattr(cov, name)
            if x is not None:
                parts.append(x[origin - spec.context_length : origin].ravel())
        if cov.known_future is not None:
            parts.append(
                (
                    cov.known_future[origin : origin + spec.max_horizon]
                    if future is None
                    else future
                ).ravel()
            )
        if cov.static is not None:
            parts.append(cov.static)
    return np.concatenate(parts)


@dataclass(frozen=True)
class DiffusionAdapter:
    module: Any
    history: Any
    spec: Any
    location: Any
    scale: Any
    # Old serialized adapters learned epsilon. Keep their interpretation when
    # unpickling objects without this field; new training sets v_prediction.
    prediction_type: str = "epsilon"

    def forecast(self, steps, covariates, *, rng: Any = None):
        v.generator(rng)
        spec = self.spec
        if steps > spec.max_horizon:
            raise ValueError("requested horizon exceeds trained diffusion horizon")
        if covariates is not None and steps != spec.max_horizon:
            raise ValueError(
                "diffusion with future covariates requires its full trained horizon"
            )
        c = condition(
            self.history, len(self.history), spec, self.location, self.scale, covariates
        )
        cond = torch.tensor(
            np.repeat(c[None], spec.samples, axis=0),
            dtype=torch.float32,
            device=spec.device,
        )
        width = self.history.dimension * spec.max_horizon
        x = torch.tensor(
            rng.normal(size=(spec.samples, width)),
            dtype=torch.float32,
            device=spec.device,
        )
        beta, alpha_bar = schedule(spec.diffusion_steps)
        with torch.no_grad():
            for t in range(spec.diffusion_steps - 1, -1, -1):
                time = torch.full(
                    (spec.samples, 1),
                    t / (spec.diffusion_steps - 1),
                    device=spec.device,
                )
                prediction = self.module(x, time, cond)
                if self.prediction_type == "v_prediction":
                    # v = sqrt(alpha_bar)*epsilon - sqrt(1-alpha_bar)*x0.
                    # Recover x0 without dividing by the near-zero terminal
                    # signal. Epsilon prediction amplifies its error there.
                    clean = (
                        np.sqrt(alpha_bar[t]) * x
                        - np.sqrt(1 - alpha_bar[t]) * prediction
                    )
                    previous_alpha = alpha_bar[t - 1] if t else 1.0
                    clean_weight = (
                        np.sqrt(previous_alpha) * beta[t] / (1 - alpha_bar[t])
                    )
                    noisy_weight = (
                        np.sqrt(1 - beta[t]) * (1 - previous_alpha) / (1 - alpha_bar[t])
                    )
                    x = clean_weight * clean + noisy_weight * x
                elif self.prediction_type == "epsilon":
                    x = (
                        x - beta[t] / np.sqrt(1 - alpha_bar[t]) * prediction
                    ) / np.sqrt(1 - beta[t])
                else:
                    raise ValueError("unknown diffusion prediction_type")
                if t:
                    variance = beta[t] * (1 - alpha_bar[t - 1]) / (1 - alpha_bar[t])
                    noise = torch.tensor(
                        rng.normal(size=(spec.samples, width)),
                        dtype=torch.float32,
                        device=spec.device,
                    )
                    x = x + np.sqrt(variance) * noise
        paths = (
            x.cpu()
            .numpy()
            .reshape(spec.samples, spec.max_horizon, self.history.dimension)
        )
        paths = paths * self.scale + self.location
        return ForecastResult.from_paths(
            paths[:, :steps],
            metadata={
                "generator": "conditional DDPM",
                "prediction_type": self.prediction_type,
                "diffusion_steps": spec.diffusion_steps,
            },
        )


def train_diffusion(spec, data, covariates):
    history = prepare(data, covariates, missing=spec.missing)
    if len(history) < spec.context_length + spec.max_horizon:
        raise ValueError("insufficient history for diffusion windows")
    location = history.values.mean(axis=0)
    scale = history.values.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    origins = range(spec.context_length, len(history) - spec.max_horizon + 1)
    conditions = np.array(
        [condition(history, o, spec, location, scale) for o in origins]
    )
    paths = np.array(
        [
            ((history.values[o : o + spec.max_horizon] - location) / scale).ravel()
            for o in origins
        ]
    )
    target = torch.tensor(paths, dtype=torch.float32, device=spec.device)
    cond = torch.tensor(conditions, dtype=torch.float32, device=spec.device)
    beta, alpha_bar = schedule(spec.diffusion_steps)
    rng = np.random.default_rng(spec.seed)
    # The built-in MLP has no dropout and initializes on CPU. All subsequent
    # random draws come from a local NumPy generator, including on CUDA.
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator().manual_seed(spec.seed).get_state())
        module = Denoiser(conditions.shape[1], paths.shape[1], spec.hidden_size).to(
            spec.device
        )
    optimizer = torch.optim.Adam(module.parameters(), lr=spec.learning_rate)
    losses = []
    for _ in range(spec.epochs):
        order = rng.permutation(len(paths))
        total, count = 0.0, 0
        for start in range(0, len(paths), spec.batch_size):
            ids = order[start : start + spec.batch_size]
            idx = torch.tensor(ids, device=spec.device)
            times = rng.integers(spec.diffusion_steps, size=len(ids))
            a = torch.tensor(
                alpha_bar[times, None], dtype=torch.float32, device=spec.device
            )
            noise = torch.tensor(
                rng.normal(size=(len(ids), paths.shape[1])),
                dtype=torch.float32,
                device=spec.device,
            )
            noisy = torch.sqrt(a) * target[idx] + torch.sqrt(1 - a) * noise
            time = torch.tensor(
                times[:, None] / (spec.diffusion_steps - 1),
                dtype=torch.float32,
                device=spec.device,
            )
            optimizer.zero_grad()
            velocity = torch.sqrt(a) * noise - torch.sqrt(1 - a) * target[idx]
            loss = ((module(noisy, time, cond[idx]) - velocity) ** 2).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite diffusion training loss")
            loss.backward()
            if any(
                p.grad is not None and not bool(torch.isfinite(p.grad).all())
                for p in module.parameters()
            ):
                raise FloatingPointError("nonfinite diffusion gradient")
            optimizer.step()
            total += float(loss.detach()) * len(ids)
            count += len(ids)
        losses.append(total / count)
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return FittedForecaster(
        DiffusionAdapter(module, history, spec, location, scale, "v_prediction"),
        history,
        spec,
        Diagnostics(
            "torch.DDPM",
            message="Fixed training budget; no optimality claim",
            iterations=spec.epochs,
            loss_history=losses,
        ),
        {
            "location": location,
            "scale": scale,
            "diffusion_steps": spec.diffusion_steps,
            "prediction_type": "v_prediction",
        },
        "diffusion.DDPM",
        spec.seed,
    )
