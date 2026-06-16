from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ai.schemas.travel import BudgetBreakdown, ItineraryDay
from ai.schemas.weather import DailyForecast, TripRisk


class TripCreate(BaseModel):
    id: UUID | None = None
    destination: str = Field(min_length=1)
    origin: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    days: int = Field(ge=1, le=30)
    travelers: int = Field(default=1, ge=1, le=20)
    status: str = "planning"
    cover_emoji: str = "\u2708\ufe0f"
    summary: str = Field(min_length=1)
    budget: BudgetBreakdown
    itinerary: list[ItineraryDay] = Field(default_factory=list)
    hotel_options: list = Field(default_factory=list)
    flight_options: list = Field(default_factory=list)
    daily_forecast: list[DailyForecast] = Field(default_factory=list)
    trip_risks: list[TripRisk] = Field(default_factory=list)
    verification_tips: list[str] = Field(default_factory=list)
    created_at: datetime | None = None


class TripResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    destination: str
    origin: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    days: int
    travelers: int
    status: str
    cover_emoji: str
    summary: str
    budget: BudgetBreakdown
    itinerary: list[ItineraryDay]
    hotel_options: list = Field(default_factory=list)
    flight_options: list = Field(default_factory=list)
    daily_forecast: list[DailyForecast] = Field(default_factory=list)
    trip_risks: list[TripRisk] = Field(default_factory=list)
    verification_tips: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
