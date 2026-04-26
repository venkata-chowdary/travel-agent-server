from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import select

from auth.models import User
from db import SessionLocal


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
