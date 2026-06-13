import logging
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from ai.agents.preference_agent import PreferenceAgent
from ai.agents.weather_agent import WeatherAgent
from ai.helpers import GeminiClient, format_preferences_block, format_weather_block
from ai.prompts import MAIN_TRAVEL_AGENT_SYSTEM_PROMPT, SUPERVISOR_ROUTING_PROMPT
from ai.schemas import PreferenceContext, TravelAgentStructuredResponse, WeatherForecastResponse
from ai.schemas.travel import TravelPlanLLMOutput
from config import settings

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

logger = logging.getLogger("travel_agent.graph")

_llm = GeminiClient(model=settings.llm_model, temperature=0)


class RoutingDecision(BaseModel):
    needs_weather: bool = Field(
        description="True whenever a specific destination is named."
    )
    destination: str | None = Field(
        default=None,
        description="City name to fetch weather for. Required when needs_weather is true.",
    )
    trip_duration_days: int = Field(
        default=3,
        description="Number of trip days. Parse from query; default 3 if unspecified.",
    )


class TravelState(TypedDict):
    user_id: str
    user_message: str
    messages: Annotated[list[BaseMessage], add_messages]
    preference_context: PreferenceContext | None
    weather_forecast: WeatherForecastResponse | None
    routing: RoutingDecision | None
    structured_response: TravelAgentStructuredResponse | None


# ── Nodes ────────────────────────────────────────────────────────────────────

async def preference_node(state: TravelState) -> dict:
    logger.info("preference_node | start | user=%s", state["user_id"])
    ctx = await PreferenceAgent().run(user_id=state["user_id"])
    logger.info("preference_node | done | home_city=%s style=%s", ctx.home_city, ctx.travel_style)
    return {"preference_context": ctx}


async def supervisor_node(state: TravelState) -> dict:
    logger.info("supervisor_node | start")
    today = datetime.now(timezone.utc).strftime("%A, %Y-%m-%d")
    routing = await _llm.with_structured_output(RoutingDecision, method="json_schema").ainvoke([
        SystemMessage(content=f"{SUPERVISOR_ROUTING_PROMPT}\n\nToday is {today}."),
        HumanMessage(content=state["user_message"]),
    ])
    logger.info(
        "supervisor_node | needs_weather=%s destination=%s days=%d",
        routing.needs_weather, routing.destination, routing.trip_duration_days,
    )
    return {"routing": routing}


async def weather_node(state: TravelState) -> dict:
    routing: RoutingDecision = state["routing"]
    today = date.today()
    trip_dates = [
        (today + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(routing.trip_duration_days or 3)
    ]
    logger.info("weather_node | start | destination=%s dates=%s", routing.destination, trip_dates)
    forecast = await WeatherAgent().run(destination=routing.destination, trip_dates=trip_dates)
    logger.info("weather_node | done | summary=%s", forecast.summary[:100])
    return {"weather_forecast": forecast}


async def main_llm_node(state: TravelState) -> dict:
    logger.info("main_llm_node | start | generating trip plan")
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

    # Build full response and inject server-side fields
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

    logger.info(
        "main_llm_node | done | destination=%s days=%d budget=%s",
        result.destination, result.days, result.budget.total,
    )
    return {"structured_response": result}


# ── Routing ──────────────────────────────────────────────────────────────────

def _route_after_supervisor(state: TravelState) -> str:
    routing = state.get("routing")
    if routing and routing.needs_weather and routing.destination:
        return "weather_node"
    return "main_llm_node"


# ── Graph ────────────────────────────────────────────────────────────────────

graph = StateGraph(TravelState)
graph.add_node("preference_node", preference_node)
graph.add_node("supervisor_node", supervisor_node)
graph.add_node("weather_node", weather_node)
graph.add_node("main_llm_node", main_llm_node)

graph.add_edge(START, "preference_node")
graph.add_edge("preference_node", "supervisor_node")
graph.add_conditional_edges("supervisor_node", _route_after_supervisor)
graph.add_edge("weather_node", "main_llm_node")
graph.add_edge("main_llm_node", END)

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
        "preference_context": None,
        "weather_forecast": None,
        "routing": None,
        "structured_response": None,
    })
    return result["structured_response"]
