from __future__ import annotations

import logging
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel

from ai.prompts import WEATHER_AGENT_SYSTEM_PROMPT
from ai.schemas.weather import WeatherForecastResponse
from ai.tools.weather_tool import get_weather_forecast

logger = logging.getLogger(__name__)

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")


def build_weather_executor(llm: BaseChatModel):
    """
    Build a LangChain 1.x agent that calls get_weather_forecast and returns
    a structured WeatherForecastResponse via response_format.
    """
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
