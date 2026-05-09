from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class FlightObjective(str, Enum):
    cheapest = "cheapest"
    fastest = "fastest"
    best_value = "best_value"


class TrainClass(str, Enum):
    sleeper = "sleeper"
    three_ac = "3ac"
    two_ac = "2ac"
    chair_car = "chair_car"


class BusType(str, Enum):
    sleeper = "sleeper"
    semi_sleeper = "semi_sleeper"
    ac = "ac"
    non_ac = "non_ac"


class HotelType(str, Enum):
    budget = "budget"
    mid_range = "mid_range"
    luxury = "luxury"


class MealType(str, Enum):
    breakfast = "breakfast"
    lunch = "lunch"
    dinner = "dinner"
    snacks = "snacks"


class ActivityInterest(str, Enum):
    adventure = "adventure"
    beach = "beach"
    culture = "culture"
    nature = "nature"
    nightlife = "nightlife"
    food = "food"
    wellness = "wellness"


class SearchResponse(BaseModel):
    success: bool = True
    count: int
    results: list[Any]


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
