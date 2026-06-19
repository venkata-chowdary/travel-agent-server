from __future__ import annotations

from pydantic import BaseModel, Field


class HotelOption(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    provider: str = Field(default="MockStay")
    area: str | None = None
    hotel_type: str = Field(min_length=1)
    price_per_night: int = Field(ge=0)
    total_price: int = Field(ge=0)
    rating: float = Field(ge=0.0, le=5.0)
    amenities: list[str] = Field(default_factory=list)
    distance_from_center_km: float | None = None
    available_rooms: int = Field(default=0, ge=0)
    refundable: bool = False
    breakfast_included: bool = False


class HotelChoiceResponse(BaseModel):
    destination: str = Field(min_length=1)
    checkin: str = Field(min_length=1)
    checkout: str = Field(min_length=1)
    nights: int = Field(ge=1)
    travelers: int = Field(default=1, ge=1)
    options: list[HotelOption] = Field(default_factory=list)
    recommended_id: str | None = None
    summary: str = Field(min_length=1)
    supervisor_note: str = Field(default="")


class HotelSelection(BaseModel):
    destination: str = Field(min_length=1)
    checkin: str = Field(min_length=1)
    checkout: str = Field(min_length=1)
    nights: int = Field(ge=1)
    travelers: int = Field(default=1, ge=1)
    selected_option: HotelOption
