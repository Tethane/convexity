"""Public roots namespace; implementation adapters remain private."""
from .builtins.roots import (
    ScalarRootSolver,
    SystemRootSolver,
    root_scalar,
    root,
)
from .builtins.fixed_point import (
    FixedPointSolver,
    fixed_point,
)

solve = root

__all__ = ['ScalarRootSolver', 'SystemRootSolver', 'root_scalar', 'root', 'FixedPointSolver', 'fixed_point', 'solve']
