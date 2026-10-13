"""Family 32: immutable fixed-deadline lifecycle and no-action bands."""

from convexity.math import policies

rule = policies.Hysteresis(entry=0.9, exit=0.6, convergence_gap=0.01)
state = policies.PolicyState(deadline=20.0)
for time, probability, gap in [
    (0.0, 0.95, 0.1),
    (5.0, 0.8, 0.06),
    (10.0, 0.82, 0.03),
    (15.0, 0.99, 0.009),
    (16.0, 1.0, 0.1),
]:
    state = rule.update(state, probability, time, gap=gap)
    print(time, state.mode, state.reason)
print("Hold wins after switching cost:", policies.no_action([5.0, 5.5], switching=0.5))
print("Mandatory override:", policies.no_action([5.0, 1.0], mandatory=1))
