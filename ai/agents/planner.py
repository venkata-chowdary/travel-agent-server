from __future__ import annotations

import logging
from datetime import datetime, timezone

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from ai.helpers import (
    format_experience_block,
    format_hotel_block,
    format_preferences_block,
    format_transport_block,
    format_weather_block,
    get_llm,
)
from ai.prompts import MAIN_TRAVEL_AGENT_SYSTEM_PROMPT
from ai.schemas import TravelAgentStructuredResponse
from ai.schemas.experience import ExperienceContext
from ai.schemas.travel import TravelPlanLLMOutput
from ai.state import TravelState, get_origin, get_trip_dates
from config import settings

logger = logging.getLogger(__name__)
_llm = get_llm(model=settings.llm_model, temperature=settings.llm_temperature)


def _experience_costs(context: ExperienceContext | None, num_travelers: int) -> tuple[int | None, int | None]:
    if context is None:
        return None, None
    activities_total = (
        sum(activity.price for activity in context.activities) * num_travelers
        if context.activities else None
    )
    food_total = (
        sum(restaurant.price_per_person for restaurant in context.restaurants) * num_travelers
        if context.restaurants else None
    )
    return activities_total, food_total


async def planner_node(state: TravelState) -> dict:
    logger.info("Planner generating trip plan...")
    num_travelers = state.get("num_travelers") or 1
    date_line = f"\nToday is {datetime.now(timezone.utc).strftime('%A, %Y-%m-%d')} (UTC)."
    travelers_line = (
        f"\nThis trip is for {num_travelers} travelers. "
        "Scale activities and food budget for the full group. "
        f"Set travelers={num_travelers} in your output."
        if num_travelers > 1 else ""
    )
    system_prompt = (
        MAIN_TRAVEL_AGENT_SYSTEM_PROMPT
        + date_line
        + travelers_line
        + format_preferences_block(state.get("preference_context"))
        + format_weather_block(state.get("weather_forecast"))
        + format_transport_block(state.get("selected_transport_options"), num_travelers)
        + format_hotel_block(state.get("selected_hotel_option"))
        + format_experience_block(state.get("experience_context"), num_travelers)
    )
    messages = [SystemMessage(content=system_prompt)]
    if state.get("messages"):
        messages.extend(state["messages"])
    messages.append(HumanMessage(content=state["user_message"]))

    try:
        plan: TravelPlanLLMOutput = await _llm.with_structured_output(
            TravelPlanLLMOutput, method="json_schema"
        ).ainvoke(messages)
    except OutputParserException as exc:
        logger.error("Planner LLM output failed schema validation: %s", exc)
        raise RuntimeError("The planner produced a malformed response. Please try again.") from exc

    try:
        result = TravelAgentStructuredResponse.model_validate(plan.model_dump())
    except ValidationError as exc:
        logger.error("Planner schema conversion failed: %s\nRaw plan: %s", exc, plan.model_dump())
        raise RuntimeError("The planner produced an invalid response. Please try again.") from exc

    updates: dict = {}
    origin = get_origin(state)
    if origin:
        updates["origin"] = origin

    trip_dates = get_trip_dates(state)
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

    selected_options = state.get("selected_transport_options")
    if selected_options:
        transport_total = sum(opt.price for opt in selected_options) * num_travelers
        result = result.model_copy(update={
            "budget": result.budget.model_copy(update={
                "flights": transport_total,
                "total": transport_total + result.budget.stay + result.budget.activities + result.budget.food,
            }),
            "transport_options": selected_options,
            "flight_options": [
                {"id": opt.id, "airline": opt.provider, "from": opt.from_, "to": opt.to,
                 "depart": opt.depart, "arrive": opt.arrive, "duration": opt.duration,
                 "price": opt.price, "stops": 0}
                for opt in selected_options if opt.mode == "flight"
            ],
        })

    selected_hotel = state.get("selected_hotel_option")
    if selected_hotel:
        result = result.model_copy(update={
            "budget": result.budget.model_copy(update={
                "stay": selected_hotel.total_price,
                "total": result.budget.flights + selected_hotel.total_price + result.budget.activities + result.budget.food,
            }),
            "hotel_options": [selected_hotel],
        })

    logger.info("Planner done — %s, %s day(s), budget %s", result.destination, result.days, result.budget.total)
    activities_total, food_total = _experience_costs(state.get("experience_context"), num_travelers)
    if activities_total is not None or food_total is not None:
        budget = result.budget.model_copy(update={
            **({"activities": activities_total} if activities_total is not None else {}),
            **({"food": food_total} if food_total is not None else {}),
        })
        result = result.model_copy(update={
            "budget": budget.model_copy(update={
                "total": budget.flights + budget.stay + budget.activities + budget.food,
            }),
        })

    return {"structured_response": result}
