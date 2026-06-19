from typing import Literal

from pydantic import BaseModel, Field, model_validator


def _normalize_legacy_origin(data):
    if isinstance(data, dict) and not data.get("origin") and data.get("home_city"):
        return {**data, "origin": data["home_city"]}
    return data


class TravelPreferences(BaseModel):
    budget_range: Literal["budget", "mid-range", "luxury"] | None = None
    travel_style: list[Literal["adventure", "relaxation", "cultural", "foodie", "family", "solo"]] = Field(default_factory=list)
    dietary_restrictions: list[str] = Field(default_factory=list)
    accommodation_type: Literal["hotel", "hostel", "airbnb", "resort", "boutique"] | None = None
    pace: Literal["relaxed", "moderate", "packed"] | None = None
    origin: str | None = None
    currency: str = "\u20b9"

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_home_city(cls, data):
        return _normalize_legacy_origin(data)


class UserProfile(BaseModel):
    name: str | None = None
    email: str | None = None


class PastTrip(BaseModel):
    destination: str
    duration_days: int
    travel_style: str
    accommodation: str
    transport: list[str] = Field(default_factory=list)
    budget_per_day_inr: float
    travel_companions: str
    pain_points: list[str] = Field(default_factory=list)
    overall_rating: float


class PreferenceContext(BaseModel):
    """Synthesized traveler context produced by the Preference Agent."""
    travel_style: str | None = None
    budget_style: Literal["budget", "mid-range", "luxury"] | None = None
    preferred_transport: list[str] = Field(default_factory=list)
    food_preference: str | None = None
    hotel_preference: str | None = None
    avoid: list[str] = Field(default_factory=list)
    memory_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    origin: str | None = None
    currency: str = "\u20b9"
    supervisor_note: str = Field(
        default="",
        description=(
            "One sentence for the supervisor: what you learned about this user and any gap "
            "that would meaningfully improve the plan if clarified before planning starts. "
            "Example: 'User prefers budget travel and dislikes crowded places \u2014 origin city is unknown, worth asking.'"
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_home_city(cls, data):
        return _normalize_legacy_origin(data)
