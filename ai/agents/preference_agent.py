from __future__ import annotations

import sys
from typing import Annotated, TypedDict
from uuid import UUID

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from ai.prompts import PREFERENCE_AGENT_SYSTEM_PROMPT
from ai.schemas import PreferenceContext
from ai.tools.preference_tools import make_preference_tools
from config import settings

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")


class _State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class PreferenceAgent:
    """
    Gathers user data via an agentic tool-calling loop, then synthesizes a
    PreferenceContext. Two Gemini calls are needed because bind_tools and
    with_structured_output cannot be combined on the same invocation.
    """

    def __init__(self) -> None:
        self._llm = ChatGoogleGenerativeAI(model=settings.llm_model, temperature=settings.llm_temperature)

    def _build_graph(self, tools: list):
        llm_with_tools = self._llm.bind_tools(tools)

        def llm_node(state: _State) -> _State:
            return {"messages": [llm_with_tools.invoke(state["messages"])]}

        graph = StateGraph(_State)
        graph.add_node("llm", llm_node)
        graph.add_node("tools", ToolNode(tools))
        graph.add_edge(START, "llm")
        graph.add_conditional_edges("llm", tools_condition)
        graph.add_edge("tools", "llm")
        return graph.compile()

    async def run(self, user_id: str | UUID) -> PreferenceContext:
        try:
            tools = make_preference_tools(user_id)
            result = await self._build_graph(tools).ainvoke({
                "messages": [
                    SystemMessage(content=PREFERENCE_AGENT_SYSTEM_PROMPT),
                    HumanMessage(content="Gather all available data about this user."),
                ]
            })

            tool_data = "\n\n".join(
                f"[{m.name}]\n{m.content}"
                for m in result["messages"]
                if isinstance(m, ToolMessage)
            )

            return await self._llm.with_structured_output(PreferenceContext).ainvoke([
                HumanMessage(content=f"{PREFERENCE_AGENT_SYSTEM_PROMPT}\n\nRaw data:\n\n{tool_data}\n\nProduce the PreferenceContext JSON.")
            ])

        except Exception:
            return PreferenceContext()
