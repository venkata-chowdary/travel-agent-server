from __future__ import annotations

from pydantic import BaseModel, Field


class ActivityOption(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    provider: str = Field(default="MockXplore")
    category: str = Field(min_length=1)
    price: int = Field(ge=0)
    duration_hours: float = Field(ge=0)
    rating: float = Field(ge=0.0, le=5.0)
    location: str | None = None
    difficulty: str | None = None
    best_time: str | None = None
    weather_sensitive: bool = False
    tags: list[str] = Field(default_factory=list)


class RestaurantOption(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    provider: str = Field(default="MockDine")
    area: str | None = None
    cuisine: str = Field(min_length=1)
    price_per_person: int = Field(ge=0)
    veg_friendly: bool = False
    rating: float = Field(ge=0.0, le=5.0)
    meal_types: list[str] = Field(default_factory=list)
    opening_hours: str | None = None
    distance_from_center_km: float | None = None
    tags: list[str] = Field(default_factory=list)


class ExperienceContext(BaseModel):
    destination: str = Field(min_length=1)
    days: int = Field(ge=1, le=30)
    travelers: int = Field(default=1, ge=1, le=20)
    activities: list[ActivityOption] = Field(default_factory=list)
    restaurants: list[RestaurantOption] = Field(default_factory=list)
    summary: str = Field(min_length=1)
    supervisor_note: str = Field(default="")
