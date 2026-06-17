from __future__ import annotations

import logging
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal, TypedDict
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from ai.agents.preference_agent import build_preference_executor
from ai.agents.transport_agent import build_transport_choice
from ai.agents.weather_agent import build_weather_executor, _unavailable_forecast
from ai.helpers import GeminiClient, format_preferences_block, format_transport_block, format_weather_block
from ai.prompts import MAIN_TRAVEL_AGENT_SYSTEM_PROMPT, SUPERVISOR_PROMPT
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

_llm = GeminiClient(model=settings.llm_model, temperature=0)


# ── Schemas ───────────────────────────────────────────────────────────────────

class SupervisorDecision(BaseModel):
    next: Literal["preference_agent", "clarifier", "weather_agent", "transport_agent", "planner"] = Field(
        description="Which agent to invoke next."
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _state_summary(state: TravelState) -> list[BaseMessage]:
    """Summarise what has been collected so far for the supervisor."""
    lines: list[str] = []
    if state.get("preference_context"):
        lines.append("- preference_context: collected")
    else:
        lines.append("- preference_context: NOT YET collected")

    if state.get("weather_forecast"):
        lines.append("- weather_forecast: collected")
    elif state.get("destination"):
        lines.append(f"- weather_forecast: NOT YET collected (destination: {state['destination']})")
    else:
        lines.append("- weather_forecast: N/A (no destination mentioned)")

    if state.get("clarification_checked"):
        lines.append("- clarification: checked")
    else:
        lines.append("- clarification: NOT YET checked")

    if state.get("selected_transport_options"):
        lines.append("- selected_transport_options: collected")
    elif state.get("transport_choice"):
        lines.append("- transport_choice: waiting for user selection")
    elif state.get("origin") and state.get("destination"):
        lines.append(f"- transport_choice: NOT YET collected (origin: {state['origin']})")
    else:
        lines.append("- transport_choice: N/A (missing origin or destination)")

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
    return prefs.home_city if prefs and prefs.home_city else None


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


def _clarification_response(state: TravelState) -> TravelAgentChatResponse | None:
    questions: list[str] = []
    origin = _origin_from_state(state)

    if not state.get("destination"):
        questions.append("Where do you want to go, or what kind of place are you in the mood for?")

    if not origin:
        questions.append("Where will you be starting from?")

    if state.get("trip_duration_days") is None:
        questions.append("How many days, or which dates, should I plan around?")

    if not questions:
        return None

    limited_questions = questions[:2]
    if len(limited_questions) == 1:
        assistant_message = f"Nice, I can plan that. Quick question first: {limited_questions[0]}"
    else:
        assistant_message = "Nice, I can shape this into a proper trip. A couple details first:\n" + "\n".join(
            f"- {question}" for question in limited_questions
        )

    return TravelAgentChatResponse(
        response_type="clarification",
        assistant_message=assistant_message,
        questions=limited_questions,
    )


# ── Nodes ────────────────────────────────────────────────────────────────────

async def supervisor_node(state: TravelState) -> dict:
    logger.info("Supervisor running — deciding next step")
    today = datetime.now(timezone.utc).strftime("%A, %Y-%m-%d")

    # Only pass history on the first run when we still need to extract destination/duration.
    # Subsequent runs (after preference_agent / weather_agent) are pure routing — _state_summary
    # already carries what was collected, no need to re-send raw history.
    needs_extraction = not state.get("destination") or state.get("trip_duration_days") is None
    history_messages = (state.get("messages") or []) if needs_extraction else []

    decision: SupervisorDecision = await _llm.with_structured_output(
        SupervisorDecision, method="json_schema"
    ).ainvoke([
        SystemMessage(content=f"{SUPERVISOR_PROMPT}\n\nToday is {today}."),
        *history_messages,
        HumanMessage(content=state["user_message"]),
        *_state_summary(state),
    ])
    logger.info(
        "Supervisor decision: %s (origin: %s, destination: %s, days: %s)",
        decision.next,
        decision.origin,
        decision.destination,
        decision.trip_duration_days,
    )
    next_step = decision.next
    previous_origin = state.get("origin") or _origin_from_state(state)
    previous_destination = state.get("destination")
    previous_days = state.get("trip_duration_days")
    previous_start_date = state.get("trip_start_date")

    origin = decision.origin or state.get("origin") or _origin_from_state(state)
    destination = decision.destination or state.get("destination")
    trip_duration_days = decision.trip_duration_days or state.get("trip_duration_days")
    trip_start_date = decision.trip_start_date or state.get("trip_start_date")

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
    weather_forecast = None if weather_invalidated else state.get("weather_forecast")
    transport_choice = None if transport_invalidated else state.get("transport_choice")
    selected_transport_options = None if transport_invalidated else state.get("selected_transport_options")

    if state.get("preference_context") and not state.get("clarification_checked"):
        next_step = "clarifier"
    elif (
        state.get("preference_context")
        and weather_forecast
        and not selected_transport_options
        and not transport_choice
        and origin
        and destination
    ):
        next_step = "transport_agent"
    elif selected_transport_options:
        next_step = "planner"
    elif weather_invalidated and destination:
        next_step = "weather_agent"

    updates = {
        "next": next_step,
        "origin": origin,
        "destination": destination,
        "trip_duration_days": trip_duration_days,
        "trip_start_date": trip_start_date,
    }
    if weather_invalidated:
        updates["weather_forecast"] = None
    if transport_invalidated:
        updates["transport_choice"] = None
        updates["selected_transport_options"] = None
    return updates


async def preference_agent_node(state: TravelState) -> dict:
    logger.info("PreferenceAgent running for user %s", state["user_id"])
    agent = build_preference_executor(state["user_id"], _llm)
    try:
        result = await agent.ainvoke({
            "messages": [("human", "Fetch all user preference data and synthesize a PreferenceContext.")]
        })
        ctx: PreferenceContext = result["structured_response"]
        logger.info("PreferenceAgent done — home city: %s, budget: %s", ctx.home_city, ctx.budget_style)
        return {"preference_context": ctx}
    except Exception:
        logger.error("PreferenceAgent failed", exc_info=True)
        return {"preference_context": PreferenceContext()}


async def clarifier_node(state: TravelState) -> dict:
    logger.info("Clarifier checking whether enough trip detail exists")
    clarification = _clarification_response(state)
    if clarification:
        logger.info("Clarifier asking %s question(s)", len(clarification.questions))
        return {
            "clarification_checked": True,
            "clarification_response": clarification,
        }

    logger.info("Clarifier passed; enough detail to plan")
    return {"clarification_checked": True}


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
        return {"weather_forecast": forecast}
    except Exception:
        logger.error("WeatherAgent failed", exc_info=True)
        return {"weather_forecast": _unavailable_forecast(destination)}


async def transport_agent_node(state: TravelState) -> dict:
    origin = _origin_from_state(state)
    destination = state.get("destination")
    trip_dates = _trip_dates(state)
    if not origin or not destination:
        logger.info("TransportAgent skipped; missing origin or destination")
        return {}

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
    return {"transport_choice": choice}


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

    result = TravelAgentStructuredResponse(**plan.model_dump())

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


# ── Routing ──────────────────────────────────────────────────────────────────

def _route(state: TravelState) -> str:
    return state["next"]


def _route_after_clarifier(state: TravelState) -> str:
    return END if state.get("clarification_response") else "supervisor"


# ── Graph ────────────────────────────────────────────────────────────────────

graph = StateGraph(TravelState)

graph.add_node("supervisor", supervisor_node)
graph.add_node("preference_agent", preference_agent_node)
graph.add_node("clarifier", clarifier_node)
graph.add_node("weather_agent", weather_agent_node)
graph.add_node("transport_agent", transport_agent_node)
graph.add_node("planner", planner_node)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("clarifier", _route_after_clarifier, {
    END: END,
    "supervisor": "supervisor",
})
graph.add_edge("preference_agent", "supervisor")
graph.add_edge("weather_agent", "supervisor")
graph.add_edge("transport_agent", END)
graph.add_conditional_edges("supervisor", _route, {
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
        import psycopg
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
