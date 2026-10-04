"""Type-only native interoperability names; importing SDK contracts loads no provider."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stable_baselines3.common.base_class import BaseAlgorithm as NativeRLModel
    from torch import Tensor as NativeTensor
