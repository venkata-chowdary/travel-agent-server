from __future__ import annotations

from datetime import date, timedelta
from typing import Literal, TypedDict, cast

from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import BaseModel, Field

from ai.schemas import (
    HotelChoiceResponse,
    HotelOption,
    PreferenceContext,
    TravelAgentChatResponse,
    TravelAgentStructuredResponse,
    TransportChoiceResponse,
    TransportOption,
    WeatherForecastResponse,
)

# ── Workflow types & constants ────────────────────────────────────────────────

WorkflowStep = Literal["preferences", "clarification", "weather", "transport", "hotel"]
WorkflowTarget = Literal["preferences", "clarification", "weather", "transport", "hotel", "planner", "none"]
WorkflowStatus = Literal["not_started", "waiting_for_user", "succeeded", "empty", "failed", "skipped_by_user"]
SupervisorIntent = Literal[
    "start_or_continue",
    "retry_step",
    "revise_details",
    "proceed_without_step",
    "select_option",
    "ask_clarification",
]

WORKFLOW_STEPS: tuple[WorkflowStep, ...] = ("preferences", "clarification", "weather", "transport", "hotel")
RESOLVED_WORKFLOW_STATUSES: set[str] = {"succeeded", "empty", "failed", "skipped_by_user"}


# ── Structured outputs the LLM fills in ──────────────────────────────────────

class ClarificationDecision(BaseModel):
    needs_clarification: bool
    questions: list[str] = Field(default_factory=list)
    assistant_message: str = ""


class SupervisorDecision(BaseModel):
    next: Literal["preference_agent", "clarifier", "weather_agent", "transport_agent", "hotel_agent", "planner"] = Field(
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
    num_travelers: int | None = Field(
        default=None,
        description=(
            "Number of people traveling (including the user). "
            "'solo' or 'just me' → 1, 'couple' or 'we two' → 2, 'family of 4' → 4. "
            "Prefer the most recent explicit value. Null if never mentioned."
        ),
    )
    companion_note: str | None = Field(
        default=None,
        description=(
            "Your brief internal observation about what you noticed in the current state "
            "that influenced this decision. E.g. 'Budget traveller asking for Maldives in peak "
            "season — worth surfacing before transport search.' Used for logging and transparency."
        ),
    )


# ── The shared state bag passed between every node ────────────────────────────

class TravelState(TypedDict):
    user_id: str
    user_message: str
    messages: list[BaseMessage]
    origin: str | None
    destination: str | None
    trip_duration_days: int | None
    trip_start_date: str | None
    num_travelers: int | None
    clarification_checked: bool
    clarification_response: TravelAgentChatResponse | None
    preference_context: PreferenceContext | None
    weather_forecast: WeatherForecastResponse | None
    weather_replan_prompted: bool
    transport_choice: TransportChoiceResponse | None
    selected_transport_options: list[TransportOption] | None
    hotel_choice: HotelChoiceResponse | None
    selected_hotel_option: HotelOption | None
    structured_response: TravelAgentStructuredResponse | None
    workflow_statuses: dict[str, WorkflowStatus]


# ── Helpers that read or derive values from TravelState ───────────────────────

def has_transport_options(choice: TransportChoiceResponse | None) -> bool:
    return bool(choice and (choice.outbound_options or choice.return_options))


def has_hotel_options(choice: HotelChoiceResponse | None) -> bool:
    return bool(choice and choice.options)


def get_origin(state: TravelState) -> str | None:
    if state.get("origin"):
        return state["origin"]
    prefs = state.get("preference_context")
    return prefs.origin if prefs and prefs.origin else None


def get_trip_dates(state: TravelState) -> list[str]:
    days = state.get("trip_duration_days") or 3
    start_str = state.get("trip_start_date")
    try:
        start = date.fromisoformat(start_str) if start_str else date.today()
    except ValueError:
        start = date.today()
    return [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]


def get_workflow_statuses(state: TravelState) -> dict[str, WorkflowStatus]:
    raw = state.get("workflow_statuses") or {}
    allowed: set[str] = {"not_started", "waiting_for_user", "succeeded", "empty", "failed", "skipped_by_user"}
    return {
        step: cast(WorkflowStatus, raw[step]) if raw.get(step) in allowed else "not_started"
        for step in WORKFLOW_STEPS
    }


def set_status(state: TravelState, step: WorkflowStep, status: WorkflowStatus) -> dict[str, WorkflowStatus]:
    statuses = get_workflow_statuses(state)
    statuses[step] = status
    return statuses


def build_state_summary(state: TravelState) -> list[BaseMessage]:
    lines: list[str] = []
    statuses = get_workflow_statuses(state)
    origin = get_origin(state)
    destination = state.get("destination")

    # ── Trip basics ───────────────────────────────────────────────────────────
    lines.append("Trip basics:")
    lines.append(f"  origin:             {origin or 'UNKNOWN'}")
    lines.append(f"  destination:        {destination or 'UNKNOWN'}")
    lines.append(f"  trip_duration_days: {state.get('trip_duration_days') or 'UNKNOWN'}")
    lines.append(f"  trip_start_date:    {state.get('trip_start_date') or 'UNKNOWN'}")
    lines.append(f"  num_travelers:      {state.get('num_travelers') or 'UNKNOWN (default 1)'}")

    # ── User profile (preference agent findings) ──────────────────────────────
    prefs = state.get("preference_context")
    if prefs:
        avoid = ", ".join(prefs.avoid) if prefs.avoid else "none"
        lines.append("\nUser profile (from preference agent):")
        lines.append(f"  travel_style:        {prefs.travel_style or 'unknown'}")
        lines.append(f"  budget_style:        {prefs.budget_style or 'unknown'}")
        lines.append(f"  preferred_transport: {', '.join(prefs.preferred_transport) if prefs.preferred_transport else 'unknown'}")
        lines.append(f"  food_preference:     {prefs.food_preference or 'unknown'}")
        lines.append(f"  hotel_preference:    {prefs.hotel_preference or 'unknown'}")
        lines.append(f"  avoid:               {avoid}")
        lines.append(f"  memory_confidence:   {prefs.memory_confidence:.2f}")
        if prefs.supervisor_note:
            lines.append(f"  → {prefs.supervisor_note}")
    else:
        lines.append("\nUser profile: NOT YET COLLECTED")

    # ── Clarification ─────────────────────────────────────────────────────────
    lines.append(f"\nClarification: {'CHECKED' if state.get('clarification_checked') else 'NOT YET CHECKED'}")

    # ── Weather findings ──────────────────────────────────────────────────────
    forecast = state.get("weather_forecast")
    if forecast:
        high_risk = [d for d in (forecast.daily_forecast or []) if d.risk_level == "high"]
        lines.append(f"\nWeather at {forecast.destination}:")
        lines.append(f"  summary:             {forecast.summary}")
        lines.append(f"  requires_replanning: {forecast.requires_replanning}")
        if high_risk:
            lines.append(f"  high-risk days:      {', '.join(d.date for d in high_risk)}")
        if forecast.supervisor_note:
            lines.append(f"  → {forecast.supervisor_note}")
    elif destination:
        lines.append(f"\nWeather: NOT YET COLLECTED (destination: {destination})")
    else:
        lines.append("\nWeather: N/A (no destination yet)")

    # ── Transport ─────────────────────────────────────────────────────────────
    if state.get("selected_transport_options"):
        lines.append("\nTransport: SELECTED BY USER")
    elif has_transport_options(state.get("transport_choice")):
        choice = state["transport_choice"]
        n = len(choice.outbound_options or []) + len(choice.return_options or [])
        lines.append(f"\nTransport: SEARCHED — {n} option(s) found (awaiting user selection)")
        if choice.supervisor_note:
            lines.append(f"  → {choice.supervisor_note}")
    elif state.get("transport_choice"):
        choice = state["transport_choice"]
        lines.append("\nTransport: SEARCHED — no options found")
        if choice.supervisor_note:
            lines.append(f"  → {choice.supervisor_note}")
    elif origin and destination:
        lines.append(f"\nTransport: NOT YET SEARCHED ({origin} → {destination})")
    else:
        lines.append("\nTransport: N/A (origin or destination unknown)")

    # ── Hotel ─────────────────────────────────────────────────────────────────
    if state.get("selected_hotel_option"):
        h = state["selected_hotel_option"]
        lines.append(f"\nHotel: SELECTED BY USER — {h.name} ({h.hotel_type}), INR {h.total_price} total")
    elif has_hotel_options(state.get("hotel_choice")):
        choice = state["hotel_choice"]
        lines.append(f"\nHotel: SEARCHED — {len(choice.options)} option(s) found (awaiting user selection)")
        if choice.supervisor_note:
            lines.append(f"  → {choice.supervisor_note}")
    elif state.get("hotel_choice"):
        lines.append("\nHotel: SEARCHED — no options found")
    elif destination:
        lines.append(f"\nHotel: NOT YET SEARCHED (destination: {destination})")
    else:
        lines.append("\nHotel: N/A (no destination yet)")

    # ── Workflow ledger ───────────────────────────────────────────────────────
    lines.append("\nWorkflow ledger:")
    for step in WORKFLOW_STEPS:
        lines.append(f"  {step}: {statuses[step]}")

    return [SystemMessage(content="Current state:\n" + "\n".join(lines))]


