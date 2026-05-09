"""
Run from the server/ directory:
    python graphs/save_graphs.py

Generates PNG snapshots of every LangGraph agent graph and saves them
under server/graphs/<phase>_<agent>.png.
"""
import sys
import os
from pathlib import Path
from typing import Annotated, TypedDict

# Make sure server/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(Path(__file__).resolve().parent.parent)

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

OUT_DIR = Path(__file__).parent


def save(graph, filename: str) -> None:
    path = OUT_DIR / filename
    img_bytes = graph.get_graph().draw_mermaid_png()
    path.write_bytes(img_bytes)
    print(f"  saved: {path.relative_to(Path(__file__).parent.parent.parent)}")


# ---------------------------------------------------------------------------
# Phase 1 — Main travel agent graph
# ---------------------------------------------------------------------------

@tool
def get_current_weather(location: str) -> str:
    """Get current weather for a location."""
    return ""

@tool
def get_current_date() -> str:
    """Get current date."""
    return ""

@tool
def get_current_time_utc() -> str:
    """Get current UTC time."""
    return ""


class _MainState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _main_agent_graph():
    main_tools = [get_current_weather, get_current_date, get_current_time_utc]

    # Placeholder node — structure only, no real LLM needed for the image
    def llm_node(state: _MainState) -> _MainState:
        return {"messages": []}

    g = StateGraph(_MainState)
    g.add_node("llm", llm_node)
    g.add_node("tools", ToolNode(main_tools))
    g.add_edge(START, "llm")
    g.add_conditional_edges("llm", tools_condition)
    g.add_edge("tools", "llm")
    return g.compile()


# ---------------------------------------------------------------------------
# Phase 1 — Preference agent graph
# ---------------------------------------------------------------------------

@tool
def get_saved_preferences() -> dict:
    """Fetch the user's saved preference settings."""
    return {}

@tool
def get_user_profile() -> dict:
    """Fetch basic user profile (name, email)."""
    return {}

@tool
def get_past_trips() -> list:
    """Fetch the user's past trip history."""
    return []


class _PrefState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _preference_agent_graph():
    pref_tools = [get_saved_preferences, get_user_profile, get_past_trips]

    def llm_node(state: _PrefState) -> _PrefState:
        return {"messages": []}

    g = StateGraph(_PrefState)
    g.add_node("llm", llm_node)
    g.add_node("tools", ToolNode(pref_tools))
    g.add_edge(START, "llm")
    g.add_conditional_edges("llm", tools_condition)
    g.add_edge("tools", "llm")
    return g.compile()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating graph images...")
    save(_main_agent_graph(),       "phase1_main_agent.png")
    save(_preference_agent_graph(), "phase1_preference_agent.png")
    print("Done.")
