from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from mock_apis.services import search_activities as _search


@tool
def search_activities_tool(
    destination: str,
    interest: str | None = None,
    max_price: float | None = None,
    duration_hours: float | None = None,
    min_rating: float | None = None,
) -> dict[str, Any]:
    """Search activities and experiences at a destination.

    Args:
        destination: City name — Goa, Bengaluru, Hyderabad, Mumbai, Delhi,
                     Jaipur, Kochi, Manali, Pondicherry, Chennai.
        interest: Category or tag to filter — 'adventure', 'beach', 'culture',
                  'nature', 'nightlife', 'food', 'wellness'.
        max_price: Maximum activity price in INR.
        duration_hours: Return only activities that finish within this many hours.
        min_rating: Minimum guest rating (0–5).

    Returns:
        Dict with keys:
          - success (bool)
          - count (int)
          - results (list): each item has id, name, category, price,
            duration_hours, rating, location, difficulty, best_time,
            weather_sensitive, tags.
    """
    return _search(
        destination=destination,
        interest=interest,
        max_price=max_price,
        duration_hours=duration_hours,
        min_rating=min_rating,
    )
