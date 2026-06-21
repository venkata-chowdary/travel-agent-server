from __future__ import annotations

import pytest

from ai.state import RESOLVED_WORKFLOW_STATUSES, SupervisorDecision
from ai.supervisor import _validate_supervisor_decision


def _statuses(
    *,
    transport_status: str = "succeeded",
    hotel_status: str = "succeeded",
    experience_status: str = "succeeded",
) -> dict[str, str]:
    return {
        "preferences": "succeeded",
        "clarification": "succeeded",
        "weather": "succeeded",
        "transport": transport_status,
        "hotel": hotel_status,
        "experience": experience_status,
    }


def test_hotel_agent_is_blocked_while_transport_awaits_selection() -> None:
    decision = SupervisorDecision(next="hotel_agent", target_step="hotel")

    issue = _validate_supervisor_decision(
        decision,
        origin="Hyderabad",
        destination="Goa",
        trip_duration_days=3,
        statuses=_statuses(transport_status="waiting_for_user", hotel_status="not_started"),
    )

    assert issue == "hotel_agent requires transport workflow to be resolved."


def test_experience_agent_is_blocked_while_transport_awaits_selection() -> None:
    decision = SupervisorDecision(next="experience_agent", target_step="experience")

    issue = _validate_supervisor_decision(
        decision,
        origin="Hyderabad",
        destination="Goa",
        trip_duration_days=3,
        statuses=_statuses(transport_status="waiting_for_user"),
    )

    assert issue == "experience_agent requires transport workflow to be resolved."


def test_experience_agent_is_blocked_while_hotel_awaits_selection() -> None:
    decision = SupervisorDecision(next="experience_agent", target_step="experience")

    issue = _validate_supervisor_decision(
        decision,
        origin="Hyderabad",
        destination="Goa",
        trip_duration_days=3,
        statuses=_statuses(hotel_status="waiting_for_user"),
    )

    assert issue == "experience_agent requires hotel workflow to be resolved."


@pytest.mark.parametrize("step", ["transport", "hotel"])
def test_planner_is_blocked_while_hitl_steps_await_selection(step: str) -> None:
    decision = SupervisorDecision(next="planner", target_step="planner")
    statuses = _statuses()
    statuses[step] = "waiting_for_user"

    issue = _validate_supervisor_decision(
        decision,
        origin="Hyderabad",
        destination="Goa",
        trip_duration_days=3,
        statuses=statuses,
    )

    assert issue is not None
    assert step in issue


def test_planner_is_blocked_until_experience_resolves() -> None:
    decision = SupervisorDecision(next="planner", target_step="planner")

    issue = _validate_supervisor_decision(
        decision,
        origin="Hyderabad",
        destination="Goa",
        trip_duration_days=3,
        statuses=_statuses(experience_status="not_started"),
    )

    assert issue is not None
    assert "experience" in issue


@pytest.mark.parametrize("experience_status", sorted(RESOLVED_WORKFLOW_STATUSES))
def test_planner_allows_resolved_experience_without_user_selection(experience_status: str) -> None:
    decision = SupervisorDecision(next="planner", target_step="planner")

    issue = _validate_supervisor_decision(
        decision,
        origin="Hyderabad",
        destination="Goa",
        trip_duration_days=3,
        statuses=_statuses(experience_status=experience_status),
    )

    assert issue is None
