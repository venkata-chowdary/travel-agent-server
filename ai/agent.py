from __future__ import annotations

import logging
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal, TypedDict, cast
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import psycopg
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError

from ai.agents.preference_agent import build_preference_executor
from ai.agents.transport_agent import build_transport_choice
from ai.agents.weather_agent import build_weather_executor, _unavailable_forecast
from ai.helpers import GeminiClient, format_preferences_block, format_transport_block, format_weather_block
from ai.prompts import CLARIFIER_PROMPT, MAIN_TRAVEL_AGENT_SYSTEM_PROMPT, SUPERVISOR_PROMPT
from ai.schemas import (
    PreferenceContext,
    TravelAgentChatResponse,
    TravelAgentStructuredResponse,
    TransportChoiceResponse,
    TransportOption,
    TransportSelection,
    WeatherForecastResponse,
)
from ai.schemas.travel import TravelPlanLLMOutput
from config import settings

logger = logging.getLogger(__name__)

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

_llm = GeminiClient(model=settings.llm_model, temperature=settings.llm_temperature)


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


# ── Schemas ───────────────────────────────────────────────────────────────────

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


# ── Helpers ───────────────────────────────────────────────────────────────────

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
            return "succeeded" if _transport_has_options(choice) else "empty"
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


def _trip_dates(state: TravelState) -> list[str]:
    days = state.get("trip_duration_days") or 3
    start_str = state.get("trip_start_date")
    try:
        start = date.fromisoformat(start_str) if start_str else date.today()
    except ValueError:
        start = date.today()
    return [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]


def _origin_from_state(state: TravelState) -> str | None:
    if state.get("origin"):
        return state["origin"]
    prefs = state.get("preference_context")
    return prefs.origin if prefs and prefs.origin else None


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


# ── Nodes ────────────────────────────────────────────────────────────────────

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
            "The previous SupervisorDecision was structurally incompatible with the workflow state. "
            f"Validation issue: {validation_issue}\n"
            "Return a corrected SupervisorDecision JSON. Do not explain."
        )))
    return messages


async def _ask_supervisor(state: TravelState, today: str, validation_issue: str | None = None) -> SupervisorDecision:
    return await _llm.with_structured_output(
        SupervisorDecision, method="json_schema"
    ).ainvoke(_supervisor_messages(state, today, validation_issue))


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
            "I need to resolve one more planning step before I can continue: "
            f"{issue}"
        ),
        questions=["How would you like me to handle that step?"],
    )


async def supervisor_node(state: TravelState) -> dict:
    logger.info("Supervisor running — deciding next step")
    today = datetime.now(timezone.utc).strftime("%A, %Y-%m-%d")
    decision = await _ask_supervisor(state, today)
    logger.info(
        "Supervisor decision: %s/%s -> %s (origin: %s, destination: %s, days: %s)",
        decision.intent,
        decision.target_step,
        decision.next,
        decision.origin,
        decision.destination,
        decision.trip_duration_days,
    )
    previous_origin = state.get("origin") or _origin_from_state(state)
    previous_destination = state.get("destination")
    previous_days = state.get("trip_duration_days")
    previous_start_date = state.get("trip_start_date")

    origin, destination, trip_duration_days, trip_start_date = _apply_decision_trip_fields(state, decision)

    origin_changed = bool(decision.origin and previous_origin and decision.origin.lower() != previous_origin.lower())
    destination_changed = bool(
        decision.destination
        and previous_destination
        and decision.destination.lower() != previous_destination.lower()
    )
    days_changed = bool(
        decision.trip_duration_days
        and previous_days
        and decision.trip_duration_days != previous_days
    )
    start_date_changed = bool(
        decision.trip_start_date
        and previous_start_date
        and decision.trip_start_date != previous_start_date
    )
    weather_invalidated = destination_changed or days_changed or start_date_changed
    transport_invalidated = origin_changed or destination_changed or days_changed or start_date_changed
    statuses = _statuses_after_intent(state, decision, weather_invalidated, transport_invalidated)
    validation_issue = _validate_supervisor_decision(decision, origin, destination, trip_duration_days, statuses)

    if validation_issue:
        logger.info("Supervisor decision incompatible; re-asking LLM: %s", validation_issue)
        decision = await _ask_supervisor(state, today, validation_issue=validation_issue)
        origin, destination, trip_duration_days, trip_start_date = _apply_decision_trip_fields(state, decision)
        origin_changed = bool(decision.origin and previous_origin and decision.origin.lower() != previous_origin.lower())
        destination_changed = bool(
            decision.destination
            and previous_destination
            and decision.destination.lower() != previous_destination.lower()
        )
        days_changed = bool(
            decision.trip_duration_days
            and previous_days
            and decision.trip_duration_days != previous_days
        )
        start_date_changed = bool(
            decision.trip_start_date
            and previous_start_date
            and decision.trip_start_date != previous_start_date
        )
        weather_invalidated = destination_changed or days_changed or start_date_changed
        transport_invalidated = origin_changed or destination_changed or days_changed or start_date_changed
        statuses = _statuses_after_intent(state, decision, weather_invalidated, transport_invalidated)
        validation_issue = _validate_supervisor_decision(decision, origin, destination, trip_duration_days, statuses)

    if validation_issue:
        return {
            "next": decision.next,
            "workflow_statuses": statuses,
            "clarification_response": _incompatible_decision_response(validation_issue),
        }

    updates = {
        "next": decision.next,
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
    return updates


async def preference_agent_node(state: TravelState) -> dict:
    logger.info("PreferenceAgent running for user %s", state["user_id"])
    agent = build_preference_executor(state["user_id"], _llm)
    try:
        result = await agent.ainvoke({
            "messages": [("human", "Fetch all user preference data and synthesize a PreferenceContext.")]
        })
        ctx: PreferenceContext = result["structured_response"]
        logger.info("PreferenceAgent done — origin: %s, budget: %s", ctx.origin, ctx.budget_style)
        return {
            "preference_context": ctx,
            "workflow_statuses": _status_update(
                state,
                "preferences",
                "succeeded" if _preference_has_data(ctx) else "empty",
            ),
        }
    except Exception:
        logger.error("PreferenceAgent failed", exc_info=True)
        return {
            "preference_context": PreferenceContext(),
            "workflow_statuses": _status_update(state, "preferences", "failed"),
        }


async def clarifier_node(state: TravelState) -> dict:
    logger.info("Clarifier running — LLM deciding whether clarification is needed")
    decision: ClarificationDecision = await _llm.with_structured_output(
        ClarificationDecision, method="json_schema"
    ).ainvoke([
        SystemMessage(content=CLARIFIER_PROMPT),
        *(state.get("messages") or []),
        HumanMessage(content=state["user_message"]),
        *_state_summary(state),
    ])

    if decision.needs_clarification:
        logger.info("Clarifier asking %s question(s)", len(decision.questions))
        return {
            "clarification_checked": True,
            "workflow_statuses": _status_update(state, "clarification", "waiting_for_user"),
            "clarification_response": TravelAgentChatResponse(
                response_type="clarification",
                assistant_message=decision.assistant_message,
                questions=decision.questions,
            ),
        }

    logger.info("Clarifier passed; enough detail to plan")
    return {
        "clarification_checked": True,
        "workflow_statuses": _status_update(state, "clarification", "succeeded"),
    }


async def weather_agent_node(state: TravelState) -> dict:
    destination = state["destination"]
    trip_dates = _trip_dates(state)
    logger.info("WeatherAgent running — %s, dates: %s", destination, trip_dates)
    agent = build_weather_executor(_llm)
    try:
        result = await agent.ainvoke({
            "messages": [("human", f"Get weather forecast for {destination} on these dates: {', '.join(trip_dates)}")]
        })
        forecast: WeatherForecastResponse = result["structured_response"]
        logger.info("WeatherAgent done — %s", forecast.summary[:80])
        return {
            "weather_forecast": forecast,
            "workflow_statuses": _status_update(
                state,
                "weather",
                "succeeded" if forecast.daily_forecast else "empty",
            ),
        }
    except Exception:
        logger.error("WeatherAgent failed", exc_info=True)
        return {
            "weather_forecast": _unavailable_forecast(destination),
            "workflow_statuses": _status_update(state, "weather", "failed"),
        }


async def transport_agent_node(state: TravelState) -> dict:
    origin = _origin_from_state(state)
    destination = state.get("destination")
    trip_dates = _trip_dates(state)
    if not origin or not destination:
        logger.info("TransportAgent skipped; missing origin or destination")
        return {"workflow_statuses": _status_update(state, "transport", "failed")}

    logger.info("TransportAgent running — %s to %s on %s", origin, destination, trip_dates[0])
    choice = build_transport_choice(
        origin=origin,
        destination=destination,
        start_date=trip_dates[0],
        days=state.get("trip_duration_days") or len(trip_dates),
        travelers=1,
        preferences=state.get("preference_context"),
    )
    logger.info(
        "TransportAgent found %s outbound and %s return option(s)",
        len(choice.outbound_options),
        len(choice.return_options),
    )
    return {
        "transport_choice": choice,
        "workflow_statuses": _status_update(
            state,
            "transport",
            "succeeded" if _transport_has_options(choice) else "empty",
        ),
    }


async def planner_node(state: TravelState) -> dict:
    logger.info("Planner generating trip plan...")
    date_line = f"\nToday is {datetime.now(timezone.utc).strftime('%A, %Y-%m-%d')} (UTC)."
    system_prompt = (
        MAIN_TRAVEL_AGENT_SYSTEM_PROMPT
        + date_line
        + format_preferences_block(state.get("preference_context"))
        + format_weather_block(state.get("weather_forecast"))
        + format_transport_block(state.get("selected_transport_options"))
    )
    messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
    if state.get("messages"):
        messages.extend(state["messages"])
    messages.append(HumanMessage(content=state["user_message"]))

    plan: TravelPlanLLMOutput = await _llm.with_structured_output(
        TravelPlanLLMOutput, method="json_schema"
    ).ainvoke(messages)

    try:
        result = TravelAgentStructuredResponse.model_validate(plan.model_dump())
    except ValidationError as exc:
        logger.error("Planner schema conversion failed: %s\nRaw plan: %s", exc, plan.model_dump())
        raise RuntimeError("The planner produced an invalid response. Please try again.") from exc

    updates: dict = {}
    origin = _origin_from_state(state)
    if origin:
        updates["origin"] = origin

    trip_dates = _trip_dates(state)
    if trip_dates:
        updates["start_date"] = trip_dates[0]
        updates["end_date"] = trip_dates[-1]

    wf = state.get("weather_forecast")
    if wf and wf.daily_forecast:
        updates["daily_forecast"] = wf.daily_forecast
        updates["trip_risks"] = wf.trip_risks
        updates["requires_replanning"] = wf.requires_replanning
        updates["weather_summary"] = wf.summary

    if updates:
        result = result.model_copy(update=updates)

    result = _apply_transport_budget(result, state.get("selected_transport_options"))

    logger.info("Planner done — %s, %s day(s), budget %s", result.destination, result.days, result.budget.total)
    return {"structured_response": result}


# ── Graph ────────────────────────────────────────────────────────────────────

graph = StateGraph(TravelState)

graph.add_node("supervisor", supervisor_node)
graph.add_node("preference_agent", preference_agent_node)
graph.add_node("clarifier", clarifier_node)
graph.add_node("weather_agent", weather_agent_node)
graph.add_node("transport_agent", transport_agent_node)
graph.add_node("planner", planner_node)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("clarifier", lambda s: END if s.get("clarification_response") else "supervisor", {
    END: END,
    "supervisor": "supervisor",
})
graph.add_edge("preference_agent", "supervisor")
graph.add_edge("weather_agent", "supervisor")
graph.add_edge("transport_agent", END)
graph.add_conditional_edges("supervisor", lambda s: END if s.get("clarification_response") else s["next"], {
    END: END,
    "preference_agent": "preference_agent",
    "clarifier": "clarifier",
    "weather_agent": "weather_agent",
    "transport_agent": "transport_agent",
    "planner": "planner",
})
graph.add_edge("planner", END)

_checkpoint_pool: Any | None = None
_checkpoint_saver: Any | None = None
agent = graph.compile()


def _checkpoint_database_url(database_url: str) -> str:
    """Return a psycopg-compatible Postgres URL for LangGraph checkpointing."""
    normalized_url = database_url.strip()
    if normalized_url.startswith("postgres://"):
        normalized_url = normalized_url.replace("postgres://", "postgresql://", 1)
    elif normalized_url.startswith("postgresql+asyncpg://"):
        normalized_url = normalized_url.replace("postgresql+asyncpg://", "postgresql://", 1)

    parsed_url = urlparse(normalized_url)
    query_params = dict(parse_qsl(parsed_url.query, keep_blank_values=True))
    ssl = query_params.pop("ssl", None)
    if ssl and "sslmode" not in query_params:
        query_params["sslmode"] = ssl

    return urlunparse(parsed_url._replace(query=urlencode(query_params)))


async def init_agent_checkpointing() -> None:
    """Initialise Postgres-backed LangGraph checkpoints for chat sessions."""
    global _checkpoint_pool, _checkpoint_saver, agent

    if _checkpoint_saver is not None:
        return

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Postgres checkpointing requires the langgraph-checkpoint-postgres package. "
            "Install backend requirements with: pip install -r requirements.txt"
        ) from exc

    checkpoint_url = _checkpoint_database_url(settings.database_url)
    serde = JsonPlusSerializer(allowed_json_modules=[
        ("ai.schemas.preferences", "PreferenceContext"),
        ("ai.schemas.transport", "TransportChoiceResponse"),
        ("ai.schemas.transport", "TransportOption"),
        ("ai.schemas.weather", "WeatherForecastResponse"),
        ("ai.schemas.weather", "DailyForecast"),
        ("ai.schemas.weather", "TripRisk"),
    ])

    _checkpoint_pool = AsyncConnectionPool(
        conninfo=checkpoint_url,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        min_size=1,
        max_size=5,
        open=False,
    )
    await _checkpoint_pool.open()

    _checkpoint_saver = AsyncPostgresSaver(conn=_checkpoint_pool, serde=serde)
    await _checkpoint_saver.setup()
    agent = graph.compile(checkpointer=_checkpoint_saver)
    logger.info("LangGraph Postgres checkpointing ready")


async def close_agent_checkpointing() -> None:
    """Close the Postgres checkpoint connection pool on app shutdown."""
    global _checkpoint_pool, _checkpoint_saver, agent

    if _checkpoint_pool is not None:
        await _checkpoint_pool.close()

    _checkpoint_pool = None
    _checkpoint_saver = None
    agent = graph.compile()


# ── Public API ───────────────────────────────────────────────────────────────

async def run_travel_agent(
    user_id: str,
    user_message: str,
    session_id: str,
    history: list[BaseMessage] | None = None,
    transport_selection: TransportSelection | None = None,
) -> TravelAgentChatResponse:
    selected_options = transport_selection.selected_options if transport_selection else None
    config: dict[str, Any] = {"configurable": {"thread_id": session_id}}

    input_state: dict[str, Any] = {
        "user_id": str(user_id),
        "user_message": user_message,
        "messages": history or [],
        "next": "",
        "clarification_checked": False,
        "clarification_response": None,
        "structured_response": None,
    }
    if transport_selection:
        input_state.update({
            "origin": transport_selection.origin,
            "destination": transport_selection.destination,
            "trip_duration_days": transport_selection.days,
            "trip_start_date": transport_selection.start_date,
            "transport_choice": None,
            "selected_transport_options": selected_options,
        })

    try:
        result = await agent.ainvoke(input_state, config=config)
    except Exception as exc:
        if isinstance(exc, psycopg.OperationalError):
            logger.warning("Postgres connection dropped, reinitialising checkpointer and retrying...")
            await close_agent_checkpointing()
            await init_agent_checkpointing()
            result = await agent.ainvoke(input_state, config=config)
        else:
            raise

    if result.get("clarification_response"):
        return result["clarification_response"]

    if result.get("transport_choice"):
        choice = result["transport_choice"]
        return TravelAgentChatResponse(
            response_type="transport_choice",
            assistant_message=choice.summary,
            transport_choice=choice,
        )

    trip_plan = result["structured_response"]
    return TravelAgentChatResponse(
        response_type="trip_plan",
        assistant_message=(
            f"Here's an idea for your {trip_plan.days}-day trip to {trip_plan.destination}."
        ),
        trip_plan=trip_plan,
    )
