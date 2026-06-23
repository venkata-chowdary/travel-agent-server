from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ai.schemas.hotel import HotelOption
from ai.schemas.travel import BudgetBreakdown, ItineraryDay
from ai.schemas.transport import TransportOption
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
    transport_options: list[TransportOption] = Field(default_factory=list)
    daily_forecast: list[DailyForecast] = Field(default_factory=list)
    trip_risks: list[TripRisk] = Field(default_factory=list)
    verification_tips: list[str] = Field(default_factory=list)
    created_at: datetime | None = None


class TripTransportOptionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    session_id: str
    trip_id: UUID | None = None
    option_id: str
    mode: str
    leg: str
    provider: str
    from_city: str
    to_city: str
    depart: str
    arrive: str
    duration: str
    price: int
    available_seats: int
    rating: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    status: str
    is_recommended: bool
    created_at: datetime


class TripHotelOptionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    session_id: str
    trip_id: UUID | None = None
    option_id: str
    destination: str
    checkin: str
    checkout: str
    nights: int
    travelers: int
    name: str
    provider: str
    area: str | None = None
    hotel_type: str
    price_per_night: int
    total_price: int
    rating: float
    amenities: list[str] = Field(default_factory=list)
    distance_from_center_km: float | None = None
    available_rooms: int
    refundable: bool
    breakfast_included: bool
    status: str
    is_recommended: bool
    created_at: datetime


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
    hotel_options: list[HotelOption] = Field(default_factory=list)
    hotel_status: str = "not_searched"
    flight_options: list = Field(default_factory=list)
    transport_options: list[TransportOption] = Field(default_factory=list)
    transport_status: str = "not_searched"
    daily_forecast: list[DailyForecast] = Field(default_factory=list)
    trip_risks: list[TripRisk] = Field(default_factory=list)
    verification_tips: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
