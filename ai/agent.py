import sys
from typing import Annotated, TypedDict
from uuid import UUID

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages

from ai.service import fetch_user_preferences
from tools.time_tool import get_current_time_utc
from tools.weather_tool import get_current_weather


load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
GENERAL_TOOLS = [get_current_time_utc, get_current_weather]
llm_with_tools = llm.bind_tools(GENERAL_TOOLS)

TRAVEL_AGENT_SYSTEM_PROMPT = """
You are a helpful travel planning assistant.

Behavior:
- Be concise, practical, and action-oriented.
- Ask brief clarifying questions when critical trip details are missing.
- Use tools only when they improve accuracy.

Tool policy:
- Use `get_current_weather` when weather affects planning or packing.
- Use `get_current_time_utc` when you need the current time reference.
- Never fabricate tool results.
""".strip()


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def llm_node(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def build_preferences_context(preferences: dict) -> str:
    if not preferences:
        return "No saved user preferences were provided."

    label_map = {"travelStyle": "Travel Style", "budgetMin": "Budget Min", "budgetMax": "Budget Max"}
    lines = [
        f"- {label_map.get(key, key)}: {value}"
        for key, value in preferences.items()
        if value not in (None, "", [], {})
    ]
    return "\n".join(lines) if lines else "No saved user preferences were provided."


def _extract_final_text(final_message: BaseMessage) -> str:
    if isinstance(final_message, AIMessage):
        return str(final_message.content)
    return str(getattr(final_message, "content", ""))


async def run_travel_agent(
    user_id: str | UUID,
    user_message: str,
    history: list[BaseMessage] | None = None,
) -> str:
    preferences = await fetch_user_preferences(user_id)
    preferences_context = build_preferences_context(preferences)

    system_message = SystemMessage(
        content=(
            f"{TRAVEL_AGENT_SYSTEM_PROMPT}\n\n"
            "Use these user preferences while planning. If the user provides new constraints in this conversation, prioritize the latest user instructions:\n"
            f"{preferences_context}"
        )
    )

    messages: list[BaseMessage] = [system_message]
    if history:
        messages.extend(history)
    messages.append(HumanMessage(content=user_message))

    state: AgentState = {"messages": messages}
    # Note: We rely on built-in tool calling from the bound model.
    response_state = llm_node(state)
    return _extract_final_text(response_state["messages"][0])