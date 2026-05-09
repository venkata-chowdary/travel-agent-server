from __future__ import annotations

from typing import Any
from uuid import UUID

from langchain_core.tools import tool

from ai.service import fetch_past_trips, fetch_user_preferences, fetch_user_profile


def make_preference_tools(user_id: str | UUID) -> list:
    """Return the three preference-gathering tools with user_id baked in via closure."""

    @tool
    async def get_saved_preferences() -> dict[str, Any]:
        """Retrieve the raw saved travel preferences for the current user.
        Returns budget_range, travel_style, dietary_restrictions, cabin_class,
        accommodation_type, pace, home_city, and currency."""
        return await fetch_user_preferences(user_id)

    @tool
    async def get_user_profile() -> dict[str, Any]:
        """Retrieve the current user's profile: name and email."""
        return await fetch_user_profile(user_id)

    @tool
    async def get_past_trips() -> list[dict[str, Any]]:
        """Retrieve the current user's past trip history.
        Each trip includes destination, duration, travel style, accommodation,
        transport, budget, and pain points. Use this to infer behavioural
        preferences the user has not explicitly stated."""
        return await fetch_past_trips(user_id)

    return [get_saved_preferences, get_user_profile, get_past_trips]
