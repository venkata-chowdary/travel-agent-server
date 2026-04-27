from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
from typing import Annotated, TypedDict
import sys
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from uuid import UUID

try:
    from ai.schemas import BudgetBreakdown, TravelAgentStructuredResponse, WeatherNotes
    from ai.service import fetch_user_preferences
    from ai.prompts import TRAVEL_AGENT_SYSTEM_PROMPT
    from tools.weather_tool import get_current_date, get_current_weather
    from app.tools.flight_tools import search_flights
except ModuleNotFoundError:
    # Allow running this module directly from `server/ai` during local debugging.
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from schemas import BudgetBreakdown, TravelAgentStructuredResponse, WeatherNotes
    from service import fetch_user_preferences
    from prompts import TRAVEL_AGENT_SYSTEM_PROMPT
    from tools.weather_tool import get_current_date, get_current_weather
    from app.tools.flight_tools import search_flights

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0,
)
tools = [get_current_date, get_current_weather, search_flights]
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


graph_builder = StateGraph(AgentState)
def llm_node(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

graph_builder.add_node("llm", llm_node)
graph_builder.add_node("tools", ToolNode(tools))

graph_builder.add_edge(START, "llm")
graph_builder.add_conditional_edges("llm", tools_condition, {"tools": "tools", "__end__": END})
graph_builder.add_edge("tools", "llm")


def build_preferences_context(preferences: dict) -> str:
    if not preferences:
        return "No saved user preferences were provided."

    label_map = {"travelStyle": "Travel Style", "budgetMin": "Budget Min", "budgetMax": "Budget Max"}
    lines = [
        f"- {label_map.get(key, key)}: {value}"
        for key, value in preferences.items()
        if value not in (None, "", [], {})
    ]

    if not lines:
        return "No saved user preferences were provided."

    return "\n".join(lines)

compiled_graph = graph_builder.compile()


def _extract_final_text(final_message: BaseMessage) -> str:
    if isinstance(final_message, AIMessage):
        return str(final_message.content)
    return str(getattr(final_message, "content", ""))


async def run_travel_agent(
    user_id: str | UUID,
    user_message: str,
    history: list[BaseMessage] | None = None,
) -> TravelAgentStructuredResponse:
    """
    Generate an assistant response using user preferences from persistence.
    """
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

    result = compiled_graph.invoke({"messages": messages}, {"recursion_limit": 8})
    final_message = result["messages"][-1]
    final_text = _extract_final_text(final_message)
    structured_llm = llm.with_structured_output(TravelAgentStructuredResponse)

    try:
        return structured_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Convert the assistant draft into a strict travel planning JSON object. "
                        "If information is missing, keep the output conservative and include actionable verification_tips."
                    )
                ),
                HumanMessage(content=final_text),
            ]
        )
    except Exception as first_error:
        try:
            return structured_llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "Repair the travel planning output to match the schema exactly. "
                            "Use conservative defaults only when necessary."
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"Validation error from previous attempt:\n{first_error}\n\n"
                            f"Original response:\n{final_text}"
                        )
                    ),
                ]
            )
        except Exception:
            return TravelAgentStructuredResponse(
                destination="Unknown destination",
                days=1,
                travelers=1,
                trip_overview=final_text
                or "I need a few more details to create a complete travel plan.",
                itinerary=[],
                budget=BudgetBreakdown(
                    flights=0,
                    stay=0,
                    activities=0,
                    food=0,
                    total=0,
                    currency="INR",
                ),
                weather=WeatherNotes(
                    summary="Weather guidance is unavailable right now. Please verify closer to travel dates.",
                    confidence="low",
                ),
                verification_tips=[
                    "Confirm destination, travel dates, and traveler count before booking.",
                    "Verify visa entry rules on the official government site.",
                    "Check weather forecast 3-5 days before departure.",
                ],
            )