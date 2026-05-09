import sys
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from ai.prompts import SYSTEM_PROMPT

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def llm_node(state: AgentState) -> AgentState:
    return {"messages": [llm.invoke(state["messages"])]}


graph = StateGraph(AgentState)
graph.add_node("llm", llm_node)
graph.add_edge(START, "llm")
graph.add_edge("llm", END)
agent = graph.compile()


async def run_travel_agent(user_message: str, history: list[BaseMessage] | None = None) -> str:
    messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    if history:
        messages.extend(history)
    messages.append(HumanMessage(content=user_message))
    result = await agent.ainvoke({"messages": messages})
    return str(result["messages"][-1].content)
