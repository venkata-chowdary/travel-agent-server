from __future__ import annotations

import logging

from langgraph.prebuilt import create_react_agent

from ai.helpers import get_llm
from ai.prompts import WEATHER_AGENT_SYSTEM_PROMPT
from ai.schemas.weather import WeatherForecastResponse
from ai.state import TravelState, get_trip_dates, set_status
from ai.tools.weather_tool import get_weather_forecast
from config import settings

logger = logging.getLogger(__name__)
_llm = get_llm(model=settings.llm_model, temperature=settings.llm_temperature)


async def weather_agent_node(state: TravelState) -> dict:
    destination = state.get("destination")
    if not destination:
        logger.warning("WeatherAgent skipped â€” destination unknown")
        return {"workflow_statuses": set_status(state, "weather", "failed")}

    trip_dates = get_trip_dates(state)
    logger.info("WeatherAgent running â€” %s, dates: %s", destination, trip_dates)

    agent = create_react_agent(
        _llm,
        [get_weather_forecast],
        prompt=WEATHER_AGENT_SYSTEM_PROMPT,
        response_format=WeatherForecastResponse,
    )

    try:
        result = await agent.ainvoke({
            "messages": [("human", f"Get weather forecast for {destination} on these dates: {', '.join(trip_dates)}")]
        })
        forecast: WeatherForecastResponse = result["structured_response"]
        status = "succeeded" if forecast.daily_forecast else "empty"
        logger.info("WeatherAgent done â€” %s", forecast.summary[:80])
        return {
            "weather_forecast": forecast,
            "workflow_statuses": set_status(state, "weather", status),
        }
    except Exception:
        logger.error("WeatherAgent failed", exc_info=True)
        return {
            "weather_forecast": WeatherForecastResponse(
                destination=destination,
                summary=f"Weather data unavailable for {destination}. Check a weather service before your trip.",
                daily_forecast=[],
                trip_risks=[],
                requires_replanning=False,
                supervisor_note=f"Weather fetch failed for {destination} â€” planner should proceed with general knowledge.",
            ),
            "workflow_statuses": set_status(state, "weather", "failed"),
        }
