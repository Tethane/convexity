"""Conditional RealNVP, sequence VAE, autoregressive Gaussian and VE score models."""
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .diffusion import condition
from .. import _validation as v
from ..interfaces.data import prepare
from ..interfaces.base import Diagnostics, FittedForecaster
from ..interfaces.result import ForecastResult


def mlp(inputs, outputs, hidden):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, outputs))


class ConditionalGenerator(nn.Module):
    def __init__(self, spec, condition_width, path_width):
        super().__init__()
        self.spec, self.width = spec, path_width
        h = spec.hidden_size
        if spec.model == 'flow':
            self.blocks = nn.ModuleList([mlp(condition_width + path_width, 2 * path_width, h) for _ in range(spec.layers)])
            self.register_buffer('masks', torch.stack([(torch.arange(path_width) + i) % 2 for i in range(spec.layers)]).float())
        elif spec.model == 'vae':
            self.encoder = mlp(condition_width + path_width, 2 * spec.latent_size, h)
            self.decoder = mlp(condition_width + spec.latent_size, 2 * path_width, h)
        elif spec.model == 'autoregressive':
            self.initial = nn.Linear(condition_width, h)
            self.cell = nn.GRUCell(1, h)
            self.emission = nn.Linear(h, 2)
        else:
            self.denoiser = mlp(condition_width + path_width + 1, path_width, h)

    def flow(self, values, context, inverse=False):
        logdet = torch.zeros(len(values), device=values.device)
        order = range(len(self.blocks) - 1, -1, -1) if inverse else range(len(self.blocks))
        for i in order:
            mask = self.get_buffer('masks')[i]
            shift, raw_scale = self.blocks[i](torch.cat((values * mask, context), dim=-1)).chunk(2, dim=-1)
            scale = 2 * torch.tanh(raw_scale / 2) * (1 - mask)
            shift = shift * (1 - mask)
            values = values * torch.exp(scale) + shift if inverse else (values - shift) * torch.exp(-scale)
            logdet = logdet + scale.sum(-1) * (1 if inverse else -1)
        return values, logdet

    def training_loss(self, target, context, rng):
        noise = lambda shape: torch.tensor(rng.normal(size=shape), dtype=target.dtype, device=target.device)
        if self.spec.model == 'flow':
            z, logdet = self.flow(target, context)
            return (.5 * (z*z + np.log(2*np.pi)).sum(-1) - logdet).mean()
        if self.spec.model == 'vae':
            mu, logvar = self.encoder(torch.cat((target, context), -1)).chunk(2, -1)
            logvar = logvar.clamp(-12, 8)
            z = mu + torch.exp(.5 * logvar) * noise(mu.shape)
            loc, scale = self.decoder(torch.cat((z, context), -1)).chunk(2, -1)
            sigma = nn.functional.softplus(scale) + 1e-4
            nll = -torch.distributions.Normal(loc, sigma).log_prob(target).sum(-1)
            kl = .5 * (mu*mu + logvar.exp() - 1 - logvar).sum(-1)
            return (nll + self.spec.kl_weight * kl).mean()
        if self.spec.model == 'autoregressive':
            hidden = torch.tanh(self.initial(context))
            loss = torch.zeros(len(target), device=target.device)
            for j in range(self.width):
                loc, raw_scale = self.emission(hidden).unbind(-1)
                loss = loss - torch.distributions.Normal(loc, nn.functional.softplus(raw_scale) + 1e-4).log_prob(target[:, j])
                hidden = self.cell(target[:, j:j+1], hidden)
            return loss.mean()
        sigma = torch.tensor(np.exp(rng.uniform(np.log(self.spec.noise_min), np.log(self.spec.noise_max), (len(target), 1))),
                             dtype=target.dtype, device=target.device)
        epsilon = noise(target.shape)
        noisy = target + sigma * epsilon
        estimate = self.denoiser(torch.cat((noisy, context, sigma.log()), -1))
        return ((estimate - epsilon)**2).mean()

    def draw(self, context, rng):
        noise = lambda width: torch.tensor(rng.normal(size=(len(context), width)), dtype=context.dtype, device=context.device)
        if self.spec.model == 'flow':
            return self.flow(noise(self.width), context, inverse=True)[0]
        if self.spec.model == 'vae':
            loc, scale = self.decoder(torch.cat((noise(self.spec.latent_size), context), -1)).chunk(2, -1)
            return loc + (nn.functional.softplus(scale) + 1e-4) * noise(self.width)
        if self.spec.model == 'autoregressive':
            hidden = torch.tanh(self.initial(context))
            values = []
            for _ in range(self.width):
                loc, scale = self.emission(hidden).chunk(2, -1)
                value = loc + (nn.functional.softplus(scale) + 1e-4) * noise(1)
                values.append(value)
                hidden = self.cell(value, hidden)
            return torch.cat(values, -1)
        values = noise(self.width) * self.spec.noise_max
        for sigma in np.geomspace(self.spec.noise_max, self.spec.noise_min, self.spec.noise_levels):
            time = torch.full((len(context), 1), float(np.log(sigma)), device=context.device)
            step = self.spec.langevin_rate * (sigma / self.spec.noise_min)**2
            for _ in range(self.spec.langevin_steps):
                score = -self.denoiser(torch.cat((values, context, time), -1)) / sigma
                values = values + step * score + np.sqrt(2 * step) * noise(self.width)
        return values


@dataclass(frozen=True)
class ConditionalGeneratorAdapter:
    module: Any
    history: Any
    spec: Any
    location: Any
    scale: Any

    def forecast(self, steps, covariates, *, rng: Any = None):
        v.generator(rng)
        if steps > self.spec.max_horizon or (covariates is not None and steps != self.spec.max_horizon):
            raise ValueError('generative horizon must fit the trained horizon and cover all future covariates')
        context = condition(self.history, len(self.history), self.spec, self.location, self.scale, covariates)
        x = torch.tensor(np.repeat(context[None], self.spec.samples, 0), dtype=torch.float32, device=self.spec.device)
        self.module.eval()
        with torch.no_grad():
            values = self.module.draw(x, rng).cpu().numpy().reshape(self.spec.samples, self.spec.max_horizon, self.history.dimension)
        return ForecastResult.from_paths((values * self.scale + self.location)[:, :steps], metadata={'generator': self.spec.model})


def train_generative(spec, data, covariates):
    history = prepare(data, covariates, missing=spec.missing)
    if len(history) < spec.context_length + spec.max_horizon:
        raise ValueError('insufficient history for generative windows')
    location, scale = history.values.mean(0), history.values.std(0)
    scale = np.where(scale > 1e-8, scale, 1.)
    origins = range(spec.context_length, len(history) - spec.max_horizon + 1)
    c = np.array([condition(history, o, spec, location, scale) for o in origins])
    y = np.array([((history.values[o:o+spec.max_horizon] - location) / scale).ravel() for o in origins])
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator().manual_seed(spec.seed).get_state())
        module = ConditionalGenerator(spec, c.shape[1], y.shape[1]).to(spec.device)
    context = torch.tensor(c, dtype=torch.float32, device=spec.device)
    target = torch.tensor(y, dtype=torch.float32, device=spec.device)
    optimizer = torch.optim.Adam(module.parameters(), lr=spec.learning_rate)
    rng, losses = np.random.default_rng(spec.seed), []
    for _ in range(spec.epochs):
        order, total = rng.permutation(len(y)), 0.
        for start in range(0, len(y), spec.batch_size):
            idx = torch.tensor(order[start:start+spec.batch_size], device=spec.device)
            optimizer.zero_grad()
            loss = module.training_loss(target[idx], context[idx], rng)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('nonfinite generative training loss')
            loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in module.parameters()):
                raise FloatingPointError('nonfinite generative gradient')
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        losses.append(total / len(y))
    module.eval()
    return FittedForecaster(ConditionalGeneratorAdapter(module, history, spec, location, scale), history, spec,
                           Diagnostics('torch.' + spec.model, message='Fixed training budget', iterations=spec.epochs,
                                       loss_history=losses), {'location': location, 'scale': scale},
                           'generative.' + spec.model, spec.seed)
