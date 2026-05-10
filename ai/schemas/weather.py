from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


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
