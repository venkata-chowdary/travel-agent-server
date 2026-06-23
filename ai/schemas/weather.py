from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DailyForecast(BaseModel):
    day: int
    date: str
    condition: str
    temperature: str
    rain_probability: int
    risk_level: Literal["low", "medium", "high"]


class TripRisk(BaseModel):
    day: int
    risk_type: str
    severity: Literal["low", "medium", "high"]
    recommendation: str


class WeatherForecastResponse(BaseModel):
    destination: str
    summary: str
    daily_forecast: list[DailyForecast]
    trip_risks: list[TripRisk]
    requires_replanning: bool
    supervisor_note: str = Field(
        default="",
        description=(
            "One sentence addressed to the supervisor. State what you found and flag "
            "any concern the planner should act on before building the itinerary. "
            "Example: '3 of 5 days show heavy rain â€” recommend alerting the user before planning outdoor activities.'"
        ),
    )
