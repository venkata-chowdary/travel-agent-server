from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import select

from ai.schemas import TravelPreferences
from auth.models import User
from db import SessionLocal


async def fetch_user_profile(user_id: str | UUID) -> dict[str, Any]:
    """Return name and email for the given user."""
    parsed_user_id = UUID(str(user_id))
    async with SessionLocal() as session:
        result = await session.execute(
            select(User.name, User.email).where(User.id == parsed_user_id)
        )
        row = result.one_or_none()
    if row is None:
        return {}
    return {"name": row.name, "email": row.email}


async def fetch_past_trips(user_id: str | UUID) -> list[dict[str, Any]]:
    """Return mock past trips. Swap body for a real DB query when trip history exists."""
    _ = user_id
    return [
        {
            "destination": "Goa",
            "duration_days": 5,
            "travel_style": "relaxation",
            "accommodation": "beachside budget stay",
            "transport": ["flight", "local taxi"],
            "budget_per_day_inr": 2500,
            "travel_companions": "solo",
            "pain_points": ["overcrowded beach", "late night bus was tiring"],
            "overall_rating": 4.5,
        },
        {
            "destination": "Coorg",
            "duration_days": 3,
            "travel_style": "relaxation",
            "accommodation": "homestay",
            "transport": ["self-drive car"],
            "budget_per_day_inr": 3000,
            "travel_companions": "couple",
            "pain_points": ["packed resort schedule felt rushed"],
            "overall_rating": 4.2,
        },
        {
            "destination": "Rajasthan",
            "duration_days": 7,
            "travel_style": "cultural",
            "accommodation": "heritage hotel",
            "transport": ["train", "auto rickshaw"],
            "budget_per_day_inr": 3500,
            "travel_companions": "family",
            "pain_points": ["itinerary too packed on days 4-5"],
            "overall_rating": 4.0,
        },
    ]


async def fetch_user_preferences(user_id: str | UUID) -> dict[str, Any]:
    """Return saved user preferences for a given user id."""
    parsed_user_id = UUID(str(user_id))

    async with SessionLocal() as session:
        result = await session.execute(
            select(User.preferences).where(User.id == parsed_user_id)
        )
        preferences = result.scalar_one_or_none()

    if isinstance(preferences, dict):
        return preferences

    return {}


async def fetch_travel_preferences(user_id: str | UUID) -> TravelPreferences:
    raw = await fetch_user_preferences(user_id)
    return TravelPreferences.model_validate(raw)
