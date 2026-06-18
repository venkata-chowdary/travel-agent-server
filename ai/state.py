from __future__ import annotations

from datetime import date, timedelta
from typing import Literal, TypedDict, cast

from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import BaseModel, Field

from ai.schemas import (
    PreferenceContext,
    TravelAgentChatResponse,
    TravelAgentStructuredResponse,
    TransportChoiceResponse,
    TransportOption,
    WeatherForecastResponse,
)

# ── Workflow types & constants ────────────────────────────────────────────────

WorkflowStep = Literal["preferences", "clarification", "weather", "transport"]
WorkflowTarget = Literal["preferences", "clarification", "weather", "transport", "planner", "none"]
WorkflowStatus = Literal["not_started", "waiting_for_user", "succeeded", "empty", "failed", "skipped_by_user"]
SupervisorIntent = Literal[
    "start_or_continue",
    "retry_step",
    "revise_details",
    "proceed_without_step",
    "select_option",
    "ask_clarification",
]

WORKFLOW_STEPS: tuple[WorkflowStep, ...] = ("preferences", "clarification", "weather", "transport")
RESOLVED_WORKFLOW_STATUSES: set[str] = {"succeeded", "empty", "failed", "skipped_by_user"}


# ── Structured outputs the LLM fills in ──────────────────────────────────────

class ClarificationDecision(BaseModel):
    needs_clarification: bool
    questions: list[str] = Field(default_factory=list)
    assistant_message: str = ""


class SupervisorDecision(BaseModel):
    next: Literal["preference_agent", "clarifier", "weather_agent", "transport_agent", "planner"] = Field(
        description="Which agent to invoke next."
    )
    intent: SupervisorIntent = Field(
        default="start_or_continue",
        description="The user's workflow intent inferred from the full conversation and current state.",
    )
    target_step: WorkflowTarget = Field(
        default="none",
        description="The workflow step the user's intent applies to, or planner/none.",
    )
    origin: str | None = Field(
        default=None,
        description="Departure city extracted from the conversation. Null if not mentioned.",
    )
    destination: str | None = Field(
        default=None,
        description="Destination city extracted from the conversation. Null if not mentioned.",
    )
    trip_duration_days: int | None = Field(
        default=None,
        description="Number of trip days parsed from the conversation. Null if not mentioned.",
    )
    trip_start_date: str | None = Field(
        default=None,
        description=(
            "ISO date (YYYY-MM-DD) of the first day of the trip, resolved from today's date. "
            "Examples: 'this weekend' → next Saturday, 'next Monday' → the coming Monday, "
            "'from the 15th' → nearest future 15th of any month. Null if no start date is mentioned."
        ),
    )


# ── The shared state bag passed between every node ────────────────────────────

class TravelState(TypedDict):
    user_id: str
    user_message: str
    messages: list[BaseMessage]
    next: str
    origin: str | None
    destination: str | None
    trip_duration_days: int | None
    trip_start_date: str | None
    clarification_checked: bool
    clarification_response: TravelAgentChatResponse | None
    preference_context: PreferenceContext | None
    weather_forecast: WeatherForecastResponse | None
    transport_choice: TransportChoiceResponse | None
    selected_transport_options: list[TransportOption] | None
    structured_response: TravelAgentStructuredResponse | None
    workflow_statuses: dict[str, WorkflowStatus]


# ── Helpers that read or derive values from TravelState ───────────────────────

def _preference_has_data(ctx: PreferenceContext | None) -> bool:
    return bool(
        ctx
        and (
            ctx.travel_style
            or ctx.budget_style
            or ctx.preferred_transport
            or ctx.food_preference
            or ctx.hotel_preference
            or ctx.avoid
            or ctx.origin
            or ctx.memory_confidence > 0
        )
    )


def _transport_has_options(choice: TransportChoiceResponse | None) -> bool:
    return bool(choice and (choice.outbound_options or choice.return_options))


def _origin_from_state(state: TravelState) -> str | None:
    if state.get("origin"):
        return state["origin"]
    prefs = state.get("preference_context")
    return prefs.origin if prefs and prefs.origin else None


def _trip_dates(state: TravelState) -> list[str]:
    days = state.get("trip_duration_days") or 3
    start_str = state.get("trip_start_date")
    try:
        start = date.fromisoformat(start_str) if start_str else date.today()
    except ValueError:
        start = date.today()
    return [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]


def _legacy_workflow_status(state: TravelState, step: WorkflowStep) -> WorkflowStatus:
    if step == "preferences":
        if state.get("preference_context"):
            return "succeeded" if _preference_has_data(state.get("preference_context")) else "empty"
        return "not_started"

    if step == "clarification":
        if state.get("clarification_response"):
            return "waiting_for_user"
        if state.get("clarification_checked"):
            return "succeeded"
        return "not_started"

    if step == "weather":
        forecast = state.get("weather_forecast")
        if forecast:
            return "succeeded" if forecast.daily_forecast else "empty"
        return "not_started"

    if step == "transport":
        if state.get("selected_transport_options"):
            return "succeeded"
        choice = state.get("transport_choice")
        if choice:
            # Options found but user hasn't selected yet — keep blocking the planner.
            return "waiting_for_user" if _transport_has_options(choice) else "empty"
        return "not_started"

    return "not_started"


def _workflow_statuses(state: TravelState) -> dict[str, WorkflowStatus]:
    raw = state.get("workflow_statuses") or {}
    statuses: dict[str, WorkflowStatus] = {}
    allowed = {"not_started", "waiting_for_user", "succeeded", "empty", "failed", "skipped_by_user"}
    for step in WORKFLOW_STEPS:
        legacy_status = _legacy_workflow_status(state, step)
        value = raw.get(step)
        if step in {"clarification", "transport"} and legacy_status in {"waiting_for_user", "succeeded"}:
            statuses[step] = legacy_status
        else:
            statuses[step] = cast(WorkflowStatus, value) if value in allowed else legacy_status
    return statuses


def _status_update(state: TravelState, step: WorkflowStep, status: WorkflowStatus) -> dict[str, WorkflowStatus]:
    statuses = _workflow_statuses(state)
    statuses[step] = status
    return statuses


def _state_summary(state: TravelState) -> list[BaseMessage]:
    lines: list[str] = []
    statuses = _workflow_statuses(state)

    if state.get("preference_context"):
        prefs = state["preference_context"]
        origin = prefs.origin or "unknown"
        lines.append(f"- preference_context: COLLECTED (origin: {origin})")
    else:
        lines.append("- preference_context: MISSING")

    if state.get("clarification_checked"):
        lines.append("- clarification: CHECKED")
    else:
        lines.append("- clarification: NOT YET CHECKED")

    origin = _origin_from_state(state)
    destination = state.get("destination")
    lines.append(f"- origin: {origin or 'UNKNOWN'}")
    lines.append(f"- destination: {destination or 'UNKNOWN'}")
    lines.append(f"- trip_duration_days: {state.get('trip_duration_days') or 'UNKNOWN'}")
    lines.append(f"- trip_start_date: {state.get('trip_start_date') or 'UNKNOWN'}")
    lines.append("- workflow ledger:")
    for step in WORKFLOW_STEPS:
        lines.append(f"  - {step}: {statuses[step]}")

    if state.get("weather_forecast"):
        lines.append("- weather_forecast: COLLECTED")
    elif destination:
        lines.append(f"- weather_forecast: NOT YET COLLECTED (destination: {destination})")
    else:
        lines.append("- weather_forecast: N/A (no destination yet)")

    if state.get("selected_transport_options"):
        lines.append("- transport: SELECTED BY USER")
    elif _transport_has_options(state.get("transport_choice")):
        lines.append("- transport: SEARCHED, OPTIONS FOUND (awaiting user selection or proceed instruction)")
    elif state.get("transport_choice"):
        lines.append("- transport: SEARCHED, NO OPTIONS FOUND")
    elif origin and destination:
        lines.append(f"- transport: NOT YET OFFERED (origin: {origin}, destination: {destination})")
    else:
        lines.append("- transport: N/A (origin or destination unknown)")

    return [SystemMessage(content="Current state:\n" + "\n".join(lines))]


def _apply_transport_budget(
    result: TravelAgentStructuredResponse,
    selected_options: list[TransportOption] | None,
) -> TravelAgentStructuredResponse:
    if not selected_options:
        return result
    transport_total = sum(option.price for option in selected_options)
    budget = result.budget.model_copy(update={
        "flights": transport_total,
        "total": transport_total + result.budget.stay + result.budget.activities + result.budget.food,
    })
    return result.model_copy(update={
        "budget": budget,
        "transport_options": selected_options,
        "flight_options": [
            _transport_option_to_flight_option(option)
            for option in selected_options
            if option.mode == "flight"
        ],
    })


def _transport_option_to_flight_option(option: TransportOption) -> dict:
    return {
        "id": option.id,
        "airline": option.provider,
        "from": option.from_,
        "to": option.to,
        "depart": option.depart,
        "arrive": option.arrive,
        "duration": option.duration,
        "price": option.price,
        "stops": 0,
    }
