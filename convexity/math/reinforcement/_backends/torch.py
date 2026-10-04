"""Differentiable actor-critic objective; target and advantage are detached."""

import torch


def actor_critic_loss(
    logits, actions, values, returns, feasible, value_weight, entropy_weight
):
    if not all(
        isinstance(x, torch.Tensor)
        for x in (logits, actions, values, returns, feasible)
    ):
        raise TypeError("actor-critic inputs must be torch tensors")
    if (
        logits.ndim != 2
        or values.shape != logits.shape[:1]
        or returns.shape != values.shape
        or actions.shape != values.shape
        or feasible.shape != logits.shape
    ):
        raise ValueError(
            "logits/mask (batch,action), actions/values/returns (batch,) required"
        )
    if (
        feasible.dtype != torch.bool
        or actions.dtype != torch.int64
        or not bool(feasible.any(dim=1).all())
    ):
        raise ValueError("boolean nonempty action masks and int64 actions required")
    if not all(bool(torch.isfinite(x).all()) for x in (logits, values, returns)):
        raise ValueError("actor-critic inputs must be finite")
    if bool(((actions < 0) | (actions >= logits.shape[1])).any()) or not bool(
        feasible.gather(1, actions[:, None]).all()
    ):
        raise ValueError("chosen actions must be feasible")
    distribution = torch.distributions.Categorical(
        logits=logits.masked_fill(~feasible, -torch.inf)
    )
    advantage = returns.detach() - values.detach()
    policy = -(distribution.log_prob(actions) * advantage).mean()
    value = 0.5 * ((values - returns.detach()) ** 2).mean()
    entropy = distribution.entropy().mean()
    return policy + value_weight * value - entropy_weight * entropy
