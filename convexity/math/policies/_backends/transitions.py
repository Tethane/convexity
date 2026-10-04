"""Priority-ordered lifecycle transitions; deterministic and side-effect free."""

from ..interfaces import PolicyState


def update(
    state, probability, time, entry, exit, convergence, adverse, mandatory, scheduled
):
    if state.mode == "closed":
        return PolicyState(state.deadline, "closed", time, state.reason)
    reason = (
        "mandatory"
        if mandatory
        else "deadline"
        if time >= state.deadline
        else "adverse_evidence"
        if adverse
        else "convergence"
        if convergence
        else "invalidation"
        if state.mode == "active" and probability <= exit
        else None
    )
    if reason is not None:
        return PolicyState(state.deadline, "closed", time, reason)
    if state.mode == "waiting" and scheduled and probability >= entry:
        return PolicyState(state.deadline, "active", time, "activation")
    return PolicyState(state.deadline, state.mode, time, "retained")
