from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt import create_react_agent

from ai.helpers import get_llm
from ai.prompts import WEATHER_AGENT_SYSTEM_PROMPT
from ai.schemas.signal import AgentSignal
from ai.schemas.weather import WeatherForecastResponse
from ai.state import TravelState, _status_update, _trip_dates
from ai.tools.weather_tool import get_weather_forecast
from config import settings

logger = logging.getLogger(__name__)
_llm = get_llm(model=settings.llm_model, temperature=settings.llm_temperature)


def build_weather_executor(llm: BaseChatModel):
    return create_react_agent(
        llm,
        [get_weather_forecast],
        prompt=WEATHER_AGENT_SYSTEM_PROMPT,
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


def _weather_signal(forecast: WeatherForecastResponse) -> AgentSignal:
    if not forecast.daily_forecast:
        return AgentSignal(
            signal_type="data_sparse",
            severity="low",
            message=f"No forecast data available for {forecast.destination}. Proceeding without weather context.",
        )
    high_risk = [d for d in forecast.daily_forecast if d.risk_level == "high"]
    medium_risk = [d for d in forecast.daily_forecast if d.risk_level == "medium"]
    total = len(forecast.daily_forecast)
    if forecast.requires_replanning:
        return AgentSignal(
            signal_type="replan_suggested",
            severity="high",
            message=(
                f"{len(high_risk)} of {total} trip days have severe weather conditions "
                f"({', '.join(d.date for d in high_risk)}). "
                "The user should know before I build a full plan around these dates."
            ),
        )
    if medium_risk:
        return AgentSignal(
            signal_type="risk_detected",
            severity="medium",
            message=(
                f"{len(medium_risk)} of {total} days have moderate weather risk. "
                "Worth mentioning, but the trip is plannable."
            ),
        )
    return AgentSignal(
        signal_type="no_action_needed",
        severity="low",
        message=f"Weather looks clear for all {total} days. No concerns.",
    )


async def weather_agent_node(state: TravelState) -> dict:
    destination = state.get("destination")
    if not destination:
        logger.warning("WeatherAgent skipped — destination unknown")
        return {
            "weather_signal": AgentSignal(
                signal_type="data_sparse",
                severity="low",
                message="Weather agent skipped — destination was not set.",
            ),
            "workflow_statuses": _status_update(state, "weather", "failed"),
        }
    trip_dates = _trip_dates(state)
    logger.info("WeatherAgent running — %s, dates: %s", destination, trip_dates)
    agent = build_weather_executor(_llm)
    try:
        result = await agent.ainvoke({
            "messages": [("human", f"Get weather forecast for {destination} on these dates: {', '.join(trip_dates)}")]
        })
        forecast: WeatherForecastResponse = result["structured_response"]
        signal = _weather_signal(forecast)
        logger.info(
            "WeatherAgent done — %s | signal: %s",
            forecast.summary[:80], signal.signal_type,
        )
        return {
            "weather_forecast": forecast,
            "weather_signal": signal,
            "workflow_statuses": _status_update(
                state, "weather", "succeeded" if forecast.daily_forecast else "empty"
            ),
        }
    except Exception:
        logger.error("WeatherAgent failed", exc_info=True)
        unavailable = _unavailable_forecast(destination)
        return {
            "weather_forecast": unavailable,
            "weather_signal": AgentSignal(
                signal_type="data_sparse",
                severity="low",
                message=f"Weather agent failed for {destination}. Proceeding without forecast.",
            ),
            "workflow_statuses": _status_update(state, "weather", "failed"),
        }
