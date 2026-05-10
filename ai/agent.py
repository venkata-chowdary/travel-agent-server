import sys
from datetime import datetime, timezone
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from ai.helpers import format_preferences_block, log_thinking
from ai.prompts import MAIN_TRAVEL_AGENT_SYSTEM_PROMPT
from ai.schemas import PreferenceContext
from config import settings
from ai.tools.weather_tool import get_current_weather

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

tools = [get_current_weather]

llm = ChatGoogleGenerativeAI(model=settings.llm_model, temperature=settings.llm_temperature)
llm_with_tools = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


async def llm_node(state: AgentState) -> AgentState:
    return {"messages": [await llm_with_tools.ainvoke(state["messages"])]}


graph = StateGraph(AgentState)

graph.add_node("llm", llm_node)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", tools_condition)
graph.add_edge("tools", "llm")
agent = graph.compile()


def _current_date_line() -> str:
    now = datetime.now(timezone.utc)
    return f"\nToday is {now.strftime('%A, %Y-%m-%d')} (UTC)."


async def run_travel_agent(
    user_message: str,
    history: list[BaseMessage] | None = None,
    preference_context: PreferenceContext | None = None,
) -> str:
    system_prompt = MAIN_TRAVEL_AGENT_SYSTEM_PROMPT + _current_date_line() + format_preferences_block(preference_context)
    messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
    if history:
        messages.extend(history)
    messages.append(HumanMessage(content=user_message))
    result = await agent.ainvoke({"messages": messages})

    # Log agent thinking — swap this for an emitter when adding frontend streaming
    log_thinking(result["messages"])

    content = result["messages"][-1].content
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("text"))
    return str(content)
