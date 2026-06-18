from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import select

from auth.models import User
from db import SessionLocal
from trips.models import Trip
from trips.service import trip_to_history


def normalize_preference_payload(preferences: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(preferences)
    if not normalized.get("origin") and normalized.get("home_city"):
        normalized["origin"] = normalized["home_city"]
    return normalized


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
    """Return persisted trip history for the given user."""
    parsed_user_id = UUID(str(user_id))
    async with SessionLocal() as session:
        result = await session.execute(
            select(Trip)
            .where(Trip.user_id == parsed_user_id)
            .order_by(Trip.created_at.desc())
            .limit(10)
        )
        trips = result.scalars().all()

    return [trip_to_history(trip) for trip in trips]


async def fetch_user_preferences(user_id: str | UUID) -> dict[str, Any]:
    """Return saved user preferences for a given user id."""
    parsed_user_id = UUID(str(user_id))

    async with SessionLocal() as session:
        result = await session.execute(
            select(User.preferences).where(User.id == parsed_user_id)
        )
        preferences = result.scalar_one_or_none()

    if isinstance(preferences, dict):
        return normalize_preference_payload(preferences)

    return {}
