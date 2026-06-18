from __future__ import annotations

import logging
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from ai.helpers import GeminiClient, format_preferences_block, format_transport_block, format_weather_block
from ai.prompts import MAIN_TRAVEL_AGENT_SYSTEM_PROMPT
from ai.schemas import TravelAgentStructuredResponse
from ai.schemas.travel import TravelPlanLLMOutput
from ai.state import TravelState, _apply_transport_budget, _origin_from_state, _trip_dates
from config import settings

logger = logging.getLogger(__name__)
_llm = GeminiClient(model=settings.llm_model, temperature=settings.llm_temperature)


async def planner_node(state: TravelState) -> dict:
    logger.info("Planner generating trip plan...")
    date_line = f"\nToday is {datetime.now(timezone.utc).strftime('%A, %Y-%m-%d')} (UTC)."
    system_prompt = (
        MAIN_TRAVEL_AGENT_SYSTEM_PROMPT
        + date_line
        + format_preferences_block(state.get("preference_context"))
        + format_weather_block(state.get("weather_forecast"))
        + format_transport_block(state.get("selected_transport_options"))
    )
    messages = [SystemMessage(content=system_prompt)]
    if state.get("messages"):
        messages.extend(state["messages"])
    messages.append(HumanMessage(content=state["user_message"]))

    plan: TravelPlanLLMOutput = await _llm.with_structured_output(
        TravelPlanLLMOutput, method="json_schema"
    ).ainvoke(messages)

    try:
        result = TravelAgentStructuredResponse.model_validate(plan.model_dump())
    except ValidationError as exc:
        logger.error("Planner schema conversion failed: %s\nRaw plan: %s", exc, plan.model_dump())
        raise RuntimeError("The planner produced an invalid response. Please try again.") from exc

    updates: dict = {}
    origin = _origin_from_state(state)
    if origin:
        updates["origin"] = origin

    trip_dates = _trip_dates(state)
    if trip_dates:
        updates["start_date"] = trip_dates[0]
        updates["end_date"] = trip_dates[-1]

    wf = state.get("weather_forecast")
    if wf and wf.daily_forecast:
        updates["daily_forecast"] = wf.daily_forecast
        updates["trip_risks"] = wf.trip_risks
        updates["requires_replanning"] = wf.requires_replanning
        updates["weather_summary"] = wf.summary

    if updates:
        result = result.model_copy(update=updates)

    result = _apply_transport_budget(result, state.get("selected_transport_options"))
    logger.info("Planner done — %s, %s day(s), budget %s", result.destination, result.days, result.budget.total)
    return {"structured_response": result}
