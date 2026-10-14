"""Explicit eligible, observable, candidate and selected research populations."""

from __future__ import annotations

from collections.abc import Mapping, Iterable

from convexity.backtest.history import HistoryError


def build_coverage(
    eligible: Mapping,
    observable: Mapping,
    *,
    failures: Iterable[dict] = (),
    candidates: Mapping | None = None,
    selected: Mapping | None = None,
) -> dict:
    """Summarize date -> universe -> security IDs without dropping missing members.

    Observable means a bar was acquired, not that a strategy has sufficient
    lookback. Pass candidates separately after validating required features.
    Unknown candidate/selected populations are reported as null, never zero.
    """
    rows = []
    for day, universes in sorted(eligible.items()):
        observed = set(observable.get(day, ()))
        for name, securities in sorted(universes.items()):
            members = set(securities)
            seen = members & observed
            candidate = (
                None if candidates is None else members & set(candidates.get(day, ()))
            )
            traded = None if selected is None else members & set(selected.get(day, ()))
            rows.append(
                dict(
                    date=str(day),
                    universe=name,
                    eligible=len(members),
                    observable=len(seen),
                    coverage_ratio=len(seen) / len(members) if members else None,
                    missing=sorted(members - seen),
                    candidates=None if candidate is None else len(candidate),
                    candidate_coverage_ratio=None
                    if candidate is None or not members
                    else len(candidate) / len(members),
                    selected=None if traded is None else len(traded),
                )
            )
    complete = all(row["eligible"] == row["observable"] for row in rows)
    return dict(
        complete=complete,
        by_date=rows,
        failures=list(failures),
        limitations=[
            "Point-in-time eligibility and observable price coverage are separate populations.",
            "Unavailable former or delisted members remain eligible and can induce survivorship/selection bias.",
            "Bar coverage does not establish sufficient strategy lookback or eliminate source membership uncertainty.",
        ],
    )


def enforce_coverage(report: dict, policy: str = "exclude") -> dict:
    """Explicitly fail incomplete research or retain its exclusion diagnostics."""
    if policy not in ("exclude", "fail"):
        raise ValueError("Missing-data policy must be 'exclude' or 'fail'.")
    if policy == "fail" and not report["complete"]:
        raise HistoryError(
            "Historical coverage is incomplete; see eligible/observable coverage report."
        )
    return report
