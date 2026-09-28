"""Explicit, immutable capability routing; no global backend state."""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping
from .. import _validation as v
from .base import NumericalTool, run


@dataclass(frozen=True)
class Backend:
    name: str
    tools: Mapping[str, NumericalTool[Any, Any]]

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("backend name must be nonempty")
        for key, tool in self.tools.items():
            if not isinstance(key, str) or not key:
                raise ValueError("capability names must be nonempty strings")
            v.function(tool, key)
        object.__setattr__(self, "tools", MappingProxyType(dict(self.tools)))

    def tool(self, capability: str):
        if capability not in self.tools:
            raise NotImplementedError(f"backend {self.name!r} does not support {capability!r}")
        return self.tools[capability]

    def run(self, capability, problem, *, check_convergence=True):
        return run(problem, self.tool(capability), check_convergence=check_convergence)

    def with_tools(self, **replacements):
        return Backend(self.name, {**self.tools, **replacements})
