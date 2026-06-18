from __future__ import annotations

import logging

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel

from ai.helpers import GeminiClient
from ai.prompts import WEATHER_AGENT_SYSTEM_PROMPT
from ai.schemas.weather import WeatherForecastResponse
from ai.state import TravelState, _status_update, _trip_dates
from ai.tools.weather_tool import get_weather_forecast
from config import settings

logger = logging.getLogger(__name__)
_llm = GeminiClient(model=settings.llm_model, temperature=settings.llm_temperature)


def build_weather_executor(llm: BaseChatModel):
    return create_agent(
        model=llm,
        tools=[get_weather_forecast],
        system_prompt=WEATHER_AGENT_SYSTEM_PROMPT,
        response_format=WeatherForecastResponse,
    )


def _unavailable_forecast(destination: str) -> WeatherForecastResponse:
    return WeatherForecastResponse(
        destination=destination,
        summary=f"Weather data unavailable for {destination}. Check a weather service before your trip.",
        daily_forecast=[],
        trip_risks=[],
        requires_replanning=False,
    )


async def weather_agent_node(state: TravelState) -> dict:
    destination = state["destination"]
    trip_dates = _trip_dates(state)
    logger.info("WeatherAgent running — %s, dates: %s", destination, trip_dates)
    agent = build_weather_executor(_llm)
    try:
        result = await agent.ainvoke({
            "messages": [("human", f"Get weather forecast for {destination} on these dates: {', '.join(trip_dates)}")]
        })
        forecast: WeatherForecastResponse = result["structured_response"]
        logger.info("WeatherAgent done — %s", forecast.summary[:80])
        return {
            "weather_forecast": forecast,
            "workflow_statuses": _status_update(
                state, "weather", "succeeded" if forecast.daily_forecast else "empty"
            ),
        }
    except Exception:
        logger.error("WeatherAgent failed", exc_info=True)
        return {
            "weather_forecast": _unavailable_forecast(destination),
            "workflow_statuses": _status_update(state, "weather", "failed"),
        }
