from typing import Literal

from pydantic import BaseModel, Field


class TravelPreferences(BaseModel):
    budget_range: Literal["budget", "mid-range", "luxury"] | None = None
    travel_style: list[Literal["adventure", "relaxation", "cultural", "foodie", "family", "solo"]] = Field(default_factory=list)
    dietary_restrictions: list[str] = Field(default_factory=list)
    accommodation_type: Literal["hotel", "hostel", "airbnb", "resort", "boutique"] | None = None
    pace: Literal["relaxed", "moderate", "packed"] | None = None
    home_city: str | None = None
    currency: str = "₹"


class PreferenceContext(BaseModel):
    """Synthesized traveler context produced by the Preference Agent."""
    travel_style: str | None = None
    budget_style: Literal["budget", "mid-range", "luxury"] | None = None
    preferred_transport: list[str] = Field(default_factory=list)
    food_preference: str | None = None
    hotel_preference: str | None = None
    avoid: list[str] = Field(default_factory=list)
    memory_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    home_city: str | None = None
    currency: str = "₹"
