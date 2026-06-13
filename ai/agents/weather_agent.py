from __future__ import annotations

import json
import logging
import sys
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from ai.helpers import GeminiClient
from ai.prompts import WEATHER_AGENT_SYSTEM_PROMPT
from ai.schemas.weather import WeatherForecastResponse
from ai.tools.weather_tool import get_weather_forecast
from config import settings

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

logger = logging.getLogger("travel_agent.weather")


class WeatherAgentState(TypedDict, total=False):
    status: str                        # "idle" | "fetched" | "synthesized" | "unavailable" | "error"
    destination: str
    trip_dates: list[str]
    raw_forecast: dict                 # raw dict returned by get_weather_forecast tool
    response: WeatherForecastResponse  # final structured output
    error: str                         # populated on failure paths


class WeatherAgent:
    """
    Fetches a multi-day weather forecast by calling the tool directly (no LLM routing),
    then synthesizes a structured WeatherForecastResponse via a single LLM call.

    Avoids LangGraph to prevent conflicts with the Google GenAI SDK's
    Automatic Function Calling (AFC), which would deadlock with a tool-calling loop.

    Execution state is tracked in self.state (WeatherAgentState) throughout run().
    """

    def __init__(self) -> None:
        self._llm = GeminiClient(model=settings.llm_model, temperature=0)
        self.state: WeatherAgentState = {"status": "idle"}

    def _unavailable(self, destination: str, reason: str) -> WeatherForecastResponse:
        return WeatherForecastResponse(
            destination=destination,
            summary=f"Weather forecast unavailable for {destination}: {reason} Please check a weather service before your trip.",
            daily_forecast=[],
            trip_risks=[],
            requires_replanning=False,
        )

    async def run(self, destination: str, trip_dates: list[str]) -> WeatherForecastResponse:
        self.state = {"status": "idle", "destination": destination, "trip_dates": trip_dates}

        logger.info("weather_agent | fetching | city=%s dates=%s", destination, trip_dates)
        raw = get_weather_forecast.invoke({"city": destination, "trip_dates": trip_dates})

        if not raw.get("days"):
            reason = raw.get("error", "No forecast data returned.")
            logger.warning("weather_agent | no forecast | reason=%s", reason)
            self.state["status"] = "unavailable"
            self.state["error"] = reason
            return self._unavailable(destination, reason)

        logger.info("weather_agent | fetched | %d day(s)", len(raw["days"]))
        self.state["raw_forecast"] = raw
        self.state["status"] = "fetched"

        raw_str = json.dumps(raw)
        dates_str = ", ".join(trip_dates)

        try:
            logger.info("weather_agent | synthesising via LLM")
            result = await self._llm.with_structured_output(WeatherForecastResponse, method="json_schema").ainvoke([
                HumanMessage(content=(
                    f"{WEATHER_AGENT_SYSTEM_PROMPT}\n\n"
                    f"Raw forecast data:\n\n{raw_str}\n\n"
                    f"Trip dates: {dates_str}\n\n"
                    "Produce the WeatherForecastResponse JSON."
                ))
            ])
            self.state["response"] = result
            self.state["status"] = "synthesized"
            logger.info("weather_agent | done | status=synthesized")
            return result
        except Exception as e:
            self.state["status"] = "error"
            self.state["error"] = "Structured synthesis failed."
            logger.error("weather_agent | synthesis failed: %s", e)
            return self._unavailable(destination, "Structured synthesis failed.")
