from __future__ import annotations

import logging
import sys
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import psycopg
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph

from ai.nodes import (
    clarifier_node,
    planner_node,
    preference_agent_node,
    transport_agent_node,
    weather_agent_node,
)
from ai.schemas import TravelAgentChatResponse, TransportSelection
from ai.state import TravelState
from ai.supervisor import supervisor_node
from config import settings

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)


# ── Graph ─────────────────────────────────────────────────────────────────────

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


# ── Checkpointing ─────────────────────────────────────────────────────────────

def _checkpoint_database_url(database_url: str) -> str:
    normalized = database_url.strip()
    if normalized.startswith("postgres://"):
        normalized = normalized.replace("postgres://", "postgresql://", 1)
    elif normalized.startswith("postgresql+asyncpg://"):
        normalized = normalized.replace("postgresql+asyncpg://", "postgresql://", 1)
    parsed = urlparse(normalized)
    params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    ssl = params.pop("ssl", None)
    if ssl and "sslmode" not in params:
        params["sslmode"] = ssl
    return urlunparse(parsed._replace(query=urlencode(params)))


async def init_agent_checkpointing() -> None:
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
            "Postgres checkpointing requires langgraph-checkpoint-postgres. "
            "Run: pip install -r requirements.txt"
        ) from exc

    serde = JsonPlusSerializer(allowed_json_modules=[
        ("ai.schemas.preferences", "PreferenceContext"),
        ("ai.schemas.transport", "TransportChoiceResponse"),
        ("ai.schemas.transport", "TransportOption"),
        ("ai.schemas.weather", "WeatherForecastResponse"),
        ("ai.schemas.weather", "DailyForecast"),
        ("ai.schemas.weather", "TripRisk"),
    ])
    _checkpoint_pool = AsyncConnectionPool(
        conninfo=_checkpoint_database_url(settings.database_url),
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
    global _checkpoint_pool, _checkpoint_saver, agent

    if _checkpoint_pool is not None:
        await _checkpoint_pool.close()
    _checkpoint_pool = None
    _checkpoint_saver = None
    agent = graph.compile()


# ── Public API ────────────────────────────────────────────────────────────────

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
            logger.warning("Postgres connection dropped — reinitialising checkpointer and retrying")
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
        assistant_message=f"Here's an idea for your {trip_plan.days}-day trip to {trip_plan.destination}.",
        trip_plan=trip_plan,
    )
