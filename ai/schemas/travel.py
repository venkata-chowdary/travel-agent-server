from typing import Literal

from pydantic import BaseModel, Field


class ItineraryItem(BaseModel):
    time: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    type: Literal["activity", "meal", "transport", "stay"]


class ItineraryDay(BaseModel):
    day: int = Field(ge=1)
    date: str | None = None
    title: str = Field(min_length=1)
    items: list[ItineraryItem] = Field(default_factory=list)


class BudgetBreakdown(BaseModel):
    flights: float = Field(ge=0)
    stay: float = Field(ge=0)
    activities: float = Field(ge=0)
    food: float = Field(ge=0)
    total: float = Field(ge=0)
    currency: str = Field(min_length=1, max_length=8)


class WeatherNotes(BaseModel):
    summary: str = Field(min_length=1)
    confidence: Literal["low", "medium", "high"] = "medium"


class TravelAgentStructuredResponse(BaseModel):
    destination: str = Field(min_length=1)
    days: int = Field(ge=1, le=30)
    travelers: int = Field(ge=1, le=20)
    trip_overview: str = Field(min_length=1)
    itinerary: list[ItineraryDay] = Field(default_factory=list)
    budget: BudgetBreakdown
    weather: WeatherNotes
    verification_tips: list[str] = Field(default_factory=list)
