"""Family 20: implied inversion, constrained smile fitting and diagnostics."""

import numpy as np
from convexity.math.pricing import forward_option
from convexity.math.implied_volatility import implied_volatility, SVI, SmileSurface

premium = forward_option(103.0, 100.0, 0.2, 1.0, discount=0.95)
print(
    "Recovered volatility:",
    implied_volatility(premium, 103.0, 100.0, 1.0, discount=0.95),
)
k = np.linspace(-1.0, 1.0, 31)
smile = SVI(0.02, 0.07, -0.3, 0.01, 0.2)
fitted = SVI.fit(k, smile(k))
surface = SmileSurface([1.0, 2.0], (fitted, SVI(0.04, 0.09, -0.3, 0.01, 0.2)))
print("Grid arbitrage checks:", surface.diagnose(k).passed)
print("18-month ATM volatility:", surface.volatility(1.5, 0.0))
