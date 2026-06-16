from __future__ import annotations

import logging
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from ai.agents.preference_agent import build_preference_executor
from ai.agents.weather_agent import build_weather_executor, _unavailable_forecast
from ai.helpers import GeminiClient, format_preferences_block, format_weather_block
from ai.prompts import MAIN_TRAVEL_AGENT_SYSTEM_PROMPT, SUPERVISOR_PROMPT
from ai.schemas import PreferenceContext, TravelAgentStructuredResponse, WeatherForecastResponse
from ai.schemas.travel import TravelPlanLLMOutput
from config import settings

logger = logging.getLogger(__name__)

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

_llm = GeminiClient(model=settings.llm_model, temperature=0)


# ── Schemas ───────────────────────────────────────────────────────────────────

class SupervisorDecision(BaseModel):
    next: Literal["preference_agent", "weather_agent", "planner"] = Field(
        description="Which agent to invoke next."
    )
    destination: str | None = Field(
        default=None,
        description="Destination city extracted from the user message. Null if not mentioned.",
    )
    trip_duration_days: int = Field(
        default=3,
        description="Number of trip days parsed from the message. Default 3.",
    )


class TravelState(TypedDict):
    user_id: str
    user_message: str
    messages: Annotated[list[BaseMessage], add_messages]
    next: str
    destination: str | None
    trip_duration_days: int
    preference_context: PreferenceContext | None
    weather_forecast: WeatherForecastResponse | None
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

    return [SystemMessage(content="Current state:\n" + "\n".join(lines))]


def _trip_dates(state: TravelState) -> list[str]:
    days = state.get("trip_duration_days") or 3
    today = date.today()
    return [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]


# ── Nodes ────────────────────────────────────────────────────────────────────

async def supervisor_node(state: TravelState) -> dict:
    logger.info("Supervisor running — deciding next step")
    today = datetime.now(timezone.utc).strftime("%A, %Y-%m-%d")
    decision: SupervisorDecision = await _llm.with_structured_output(
        SupervisorDecision, method="json_schema"
    ).ainvoke([
        SystemMessage(content=f"{SUPERVISOR_PROMPT}\n\nToday is {today}."),
        HumanMessage(content=state["user_message"]),
        *_state_summary(state),
    ])
    logger.info("Supervisor → %s (destination: %s, days: %s)",
                decision.next, decision.destination, decision.trip_duration_days)
    return {
        "next": decision.next,
        "destination": decision.destination or state.get("destination"),
        "trip_duration_days": decision.trip_duration_days or state.get("trip_duration_days") or 3,
    }


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


async def planner_node(state: TravelState) -> dict:
    logger.info("Planner generating trip plan...")
    date_line = f"\nToday is {datetime.now(timezone.utc).strftime('%A, %Y-%m-%d')} (UTC)."
    system_prompt = (
        MAIN_TRAVEL_AGENT_SYSTEM_PROMPT
        + date_line
        + format_preferences_block(state.get("preference_context"))
        + format_weather_block(state.get("weather_forecast"))
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
    if state.get("preference_context"):
        updates["origin"] = state["preference_context"].home_city

    wf = state.get("weather_forecast")
    if wf and wf.daily_forecast:
        updates["daily_forecast"] = wf.daily_forecast
        updates["trip_risks"] = wf.trip_risks
        updates["requires_replanning"] = wf.requires_replanning
        updates["weather_summary"] = wf.summary

    if updates:
        result = result.model_copy(update=updates)

    logger.info("Planner done — %s, %s day(s), budget %s", result.destination, result.days, result.budget.total)
    return {"structured_response": result}


# ── Routing ──────────────────────────────────────────────────────────────────

def _route(state: TravelState) -> str:
    return state["next"]


# ── Graph ────────────────────────────────────────────────────────────────────

graph = StateGraph(TravelState)

graph.add_node("supervisor", supervisor_node)
graph.add_node("preference_agent", preference_agent_node)
graph.add_node("weather_agent", weather_agent_node)
graph.add_node("planner", planner_node)

graph.add_edge(START, "supervisor")
graph.add_edge("preference_agent", "supervisor")   # loop back → supervisor decides next
graph.add_edge("weather_agent", "supervisor")      # loop back → supervisor decides next
graph.add_conditional_edges("supervisor", _route, {
    "preference_agent": "preference_agent",
    "weather_agent": "weather_agent",
    "planner": "planner",
})
graph.add_edge("planner", END)

agent = graph.compile()


# ── Public API ───────────────────────────────────────────────────────────────

async def run_travel_agent(
    user_id: str,
    user_message: str,
    history: list[BaseMessage] | None = None,
) -> TravelAgentStructuredResponse:
    result = await agent.ainvoke({
        "user_id": user_id,
        "user_message": user_message,
        "messages": history or [],
        "next": "",
        "destination": None,
        "trip_duration_days": 3,
        "preference_context": None,
        "weather_forecast": None,
        "structured_response": None,
    })
    return result["structured_response"]
