"""Lazy PyTorch access. Tensor parameters/results never pass through NumPy."""


def torch_module():
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        raise ImportError("Tensor distributions require: pip install 'convexity[autodiff]'") from exc
    return torch


def make(name, *args, **kwargs):
    torch = torch_module()
    return getattr(torch.distributions, name)(*args, validate_args=True, **kwargs)
