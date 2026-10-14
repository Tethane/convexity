"""cross-Greeks and exposure-neutral constrained holdings."""

from convexity.math.sensitivities import bump_sensitivities, hedge

result = bump_sensitivities(
    lambda x: x[0] ** 2 + 3 * x[0] * x[1] + 2 * x[1] ** 2, [2.0, 3.0], steps=1e-3
)
print("Hessian:", result["hessian"])
hedged = hedge(
    [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
    [2.0, -1.0],
    equality=[[1.0, 1.0, 1.0]],
    equality_target=[0.0],
)
print("Holdings:", hedged.holdings, "Residual:", hedged.residual_exposure)
