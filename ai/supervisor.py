from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal, cast

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.types import Command

from ai.helpers import get_llm
from ai.prompts import SUPERVISOR_PROMPT
from ai.schemas import TravelAgentChatResponse
from ai.state import (
    RESOLVED_WORKFLOW_STATUSES,
    WORKFLOW_STEPS,
    SupervisorDecision,
    TravelState,
    WorkflowStep,
    WorkflowStatus,
    _origin_from_state,
    _state_summary,
    _status_update,
    _transport_has_options,
    _workflow_statuses,
)
from config import settings

logger = logging.getLogger(__name__)
_llm = get_llm(model=settings.llm_model, temperature=settings.llm_temperature)


# ── Private helpers ───────────────────────────────────────────────────────────

def _supervisor_messages(
    state: TravelState,
    today: str,
    validation_issue: str | None = None,
) -> list[BaseMessage]:
    needs_extraction = (
        not _origin_from_state(state)
        or not state.get("destination")
        or state.get("trip_duration_days") is None
    )
    history_messages = (state.get("messages") or []) if needs_extraction else []
    messages: list[BaseMessage] = [
        SystemMessage(content=f"{SUPERVISOR_PROMPT}\n\nToday is {today}."),
        *history_messages,
        HumanMessage(content=state["user_message"]),
        *_state_summary(state),
    ]
    if validation_issue:
        messages.append(SystemMessage(content=(
            "The previous SupervisorDecision was invalid. "
            f"Issue: {validation_issue}\n"
            "Return a corrected SupervisorDecision JSON. Do not explain."
        )))
    return messages


async def _ask_supervisor(
    state: TravelState, today: str, validation_issue: str | None = None
) -> SupervisorDecision:
    # json_schema → Gemini controlled generation: token probabilities constrained
    # to only valid Literal enum values. OutputParserException here means a genuine
    # API failure (not a hallucination), so surface it as ValueError for retry.
    try:
        return await _llm.with_structured_output(
            SupervisorDecision, method="json_schema"
        ).ainvoke(_supervisor_messages(state, today, validation_issue))
    except OutputParserException as exc:
        raise ValueError(str(exc)) from exc


def _apply_decision_trip_fields(
    state: TravelState,
    decision: SupervisorDecision,
) -> tuple[str | None, str | None, int | None, str | None]:
    return (
        decision.origin or state.get("origin") or _origin_from_state(state),
        decision.destination or state.get("destination"),
        decision.trip_duration_days or state.get("trip_duration_days"),
        decision.trip_start_date or state.get("trip_start_date"),
    )


def _agent_for_step(step: str) -> str | None:
    return {
        "preferences": "preference_agent",
        "clarification": "clarifier",
        "weather": "weather_agent",
        "transport": "transport_agent",
    }.get(step)


def _statuses_after_intent(
    state: TravelState,
    decision: SupervisorDecision,
    weather_invalidated: bool,
    transport_invalidated: bool,
) -> dict[str, WorkflowStatus]:
    statuses = _workflow_statuses(state)

    if state.get("selected_transport_options") and statuses.get("transport") in ("waiting_for_user", "not_started"):
        statuses["transport"] = "succeeded"

    if weather_invalidated:
        statuses["weather"] = "not_started"
    if transport_invalidated:
        statuses["transport"] = "not_started"

    if decision.target_step in WORKFLOW_STEPS:
        target = cast(WorkflowStep, decision.target_step)
        if decision.intent == "retry_step":
            statuses[target] = "not_started"
        elif decision.intent == "proceed_without_step":
            statuses[target] = "skipped_by_user"
    return statuses


def _validate_supervisor_decision(
    decision: SupervisorDecision,
    origin: str | None,
    destination: str | None,
    trip_duration_days: int | None,
    statuses: dict[str, WorkflowStatus],
) -> str | None:
    if decision.intent == "retry_step":
        expected_agent = _agent_for_step(decision.target_step)
        if expected_agent is None:
            return "retry_step intent must target one specialist workflow step."
        if decision.next != expected_agent:
            return f"retry_step for {decision.target_step} must route next to {expected_agent}."

    if decision.next == "weather_agent":
        if not destination:
            return "weather_agent requires a known destination."
        if trip_duration_days is None:
            return "weather_agent requires a known trip duration."

    if decision.next == "transport_agent":
        if not origin or not destination:
            return "transport_agent requires known origin and destination."
        if trip_duration_days is None:
            return "transport_agent requires a known trip duration."

    if decision.next == "planner":
        if not origin or not destination or trip_duration_days is None:
            return "planner requires origin, destination, and trip duration."
        unresolved = [
            step for step, status in statuses.items()
            if step in WORKFLOW_STEPS and status not in RESOLVED_WORKFLOW_STATUSES
        ]
        if unresolved:
            return (
                "planner requires all specialist workflow steps to be resolved. "
                f"Unresolved steps: {', '.join(unresolved)}."
            )

    return None


def _incompatible_decision_response(issue: str) -> TravelAgentChatResponse:
    return TravelAgentChatResponse(
        response_type="clarification",
        assistant_message=(
            f"I need to resolve one more planning step before I can continue: {issue}"
        ),
        questions=["How would you like me to handle that step?"],
    )


_SupervisorDest = Literal["preference_agent", "clarifier", "weather_agent", "transport_agent", "planner"]


# ── Node ──────────────────────────────────────────────────────────────────────

async def supervisor_node(state: TravelState) -> Command[_SupervisorDest]:
    logger.info("Supervisor running — deciding next step")
    today = datetime.now(timezone.utc).strftime("%A, %Y-%m-%d")

    try:
        decision = await _ask_supervisor(state, today)
    except ValueError as exc:
        logger.warning("Supervisor parse failed; retrying with correction: %s", exc)
        try:
            decision = await _ask_supervisor(state, today, validation_issue=str(exc))
        except ValueError as exc2:
            logger.error("Supervisor retry also failed: %s", exc2)
            return Command(goto=END, update={
                "clarification_response": _incompatible_decision_response(str(exc2)),
            })

    logger.info(
        "Supervisor decision: %s/%s -> %s (origin: %s, destination: %s, days: %s, start: %s)",
        decision.intent, decision.target_step, decision.next,
        decision.origin, decision.destination, decision.trip_duration_days, decision.trip_start_date,
    )
    if decision.companion_note:
        logger.info("Companion reasoning: %s", decision.companion_note)

    previous_origin = state.get("origin") or _origin_from_state(state)
    previous_destination = state.get("destination")
    previous_days = state.get("trip_duration_days")
    previous_start_date = state.get("trip_start_date")

    origin, destination, trip_duration_days, trip_start_date = _apply_decision_trip_fields(state, decision)

    origin_changed = bool(decision.origin and previous_origin and decision.origin.lower() != previous_origin.lower())
    destination_changed = bool(decision.destination and previous_destination and decision.destination.lower() != previous_destination.lower())
    days_changed = bool(decision.trip_duration_days and previous_days and decision.trip_duration_days != previous_days)
    start_date_changed = bool(decision.trip_start_date and previous_start_date and decision.trip_start_date != previous_start_date)

    weather_invalidated = destination_changed or days_changed or start_date_changed
    transport_invalidated = origin_changed or destination_changed or days_changed or start_date_changed

    statuses = _statuses_after_intent(state, decision, weather_invalidated, transport_invalidated)
    validation_issue = _validate_supervisor_decision(decision, origin, destination, trip_duration_days, statuses)

    if validation_issue:
        if (
            statuses.get("transport") == "waiting_for_user"
            and _transport_has_options(state.get("transport_choice"))
            and not state.get("selected_transport_options")
        ):
            logger.info("Supervisor: transport options pending user selection — re-routing to transport_agent")
            return Command(goto="transport_agent", update={
                "origin": origin,
                "destination": destination,
                "trip_duration_days": trip_duration_days,
                "trip_start_date": trip_start_date,
                "workflow_statuses": {**statuses, "transport": "not_started"},
            })

        logger.info("Supervisor decision incompatible; re-asking LLM: %s", validation_issue)
        try:
            decision = await _ask_supervisor(state, today, validation_issue=validation_issue)
        except ValueError as exc:
            logger.error("Supervisor business-logic retry failed: %s", exc)
            return Command(goto=END, update={
                "workflow_statuses": statuses,
                "clarification_response": _incompatible_decision_response(str(exc)),
            })

        origin, destination, trip_duration_days, trip_start_date = _apply_decision_trip_fields(state, decision)

        origin_changed = bool(decision.origin and previous_origin and decision.origin.lower() != previous_origin.lower())
        destination_changed = bool(decision.destination and previous_destination and decision.destination.lower() != previous_destination.lower())
        days_changed = bool(decision.trip_duration_days and previous_days and decision.trip_duration_days != previous_days)
        start_date_changed = bool(decision.trip_start_date and previous_start_date and decision.trip_start_date != previous_start_date)

        weather_invalidated = destination_changed or days_changed or start_date_changed
        transport_invalidated = origin_changed or destination_changed or days_changed or start_date_changed

        statuses = _statuses_after_intent(state, decision, weather_invalidated, transport_invalidated)
        validation_issue = _validate_supervisor_decision(decision, origin, destination, trip_duration_days, statuses)

    if validation_issue:
        return Command(goto=END, update={
            "workflow_statuses": statuses,
            "clarification_response": _incompatible_decision_response(validation_issue),
        })

    updates: dict = {
        "origin": origin,
        "destination": destination,
        "trip_duration_days": trip_duration_days,
        "trip_start_date": trip_start_date,
        "workflow_statuses": statuses,
    }
    if weather_invalidated:
        updates["weather_forecast"] = None
    if transport_invalidated:
        updates["transport_choice"] = None
        updates["selected_transport_options"] = None
    if decision.intent == "retry_step":
        if decision.target_step == "preferences":
            updates["preference_context"] = None
        elif decision.target_step == "weather":
            updates["weather_forecast"] = None
        elif decision.target_step == "transport":
            updates["transport_choice"] = None
            updates["selected_transport_options"] = None
    if decision.intent == "proceed_without_step" and decision.target_step == "transport":
        updates["transport_choice"] = None
        updates["selected_transport_options"] = None
    if decision.next == "planner" and not state.get("selected_transport_options"):
        updates["transport_choice"] = None
    return Command(goto=decision.next, update=updates)
