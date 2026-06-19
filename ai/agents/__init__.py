# Agent implementation patterns used in this package:
#
#   create_react_agent (langgraph.prebuilt)
#     Use when the agent must call tools to gather data before synthesising output.
#     The ReAct loop lets the LLM decide how to use its tools and reason over results.
#     Examples: preference_agent (DB lookups), weather_agent (external API).
#
#   Direct LLM + .with_structured_output()
#     Use when the agent needs one focused LLM call with known inputs and no tool loop.
#     Examples: clarifier, planner, supervisor.
#
#   Pure Python (no LLM)
#     Use when the logic is fully deterministic given the graph state.
#     Example: transport_agent (mock API search + ranking).

from .clarifier import clarifier_node
from .hotel_agent import build_hotel_choice, hotel_agent_node
from .planner import planner_node
from .preference_agent import preference_agent_node
from .transport_agent import build_transport_choice, transport_agent_node
from .weather_agent import weather_agent_node

__all__ = [
    "build_hotel_choice",
    "build_transport_choice",
    "clarifier_node",
    "hotel_agent_node",
    "planner_node",
    "preference_agent_node",
    "transport_agent_node",
    "weather_agent_node",
]
