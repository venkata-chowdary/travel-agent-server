from __future__ import annotations

from ai.state import SupervisorDecision
from ai.supervisor import _validate_supervisor_decision


def _statuses(experience_status: str) -> dict[str, str]:
    return {
        "preferences": "succeeded",
        "clarification": "succeeded",
        "weather": "succeeded",
        "transport": "succeeded",
        "hotel": "succeeded",
        "experience": experience_status,
    }


def test_planner_is_blocked_until_experience_resolves() -> None:
    decision = SupervisorDecision(next="planner", target_step="planner")

    issue = _validate_supervisor_decision(
        decision,
        origin="Hyderabad",
        destination="Goa",
        trip_duration_days=3,
        statuses=_statuses("not_started"),
    )

    assert issue is not None
    assert "experience" in issue


def test_planner_allows_resolved_empty_experience() -> None:
    decision = SupervisorDecision(next="planner", target_step="planner")

    issue = _validate_supervisor_decision(
        decision,
        origin="Hyderabad",
        destination="Goa",
        trip_duration_days=3,
        statuses=_statuses("empty"),
    )

    assert issue is None
