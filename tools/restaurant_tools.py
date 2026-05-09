from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from mock_apis.services import search_restaurants as _search


@tool
def search_restaurants_tool(
    destination: str,
    cuisine: str | None = None,
    max_price_per_person: float | None = None,
    meal_type: str | None = None,
    veg_only: bool | None = None,
    min_rating: float | None = None,
) -> dict[str, Any]:
    """Search restaurants at a destination.

    Args:
        destination: City name — Goa, Bengaluru, Hyderabad, Mumbai, Delhi,
                     Jaipur, Kochi, Manali, Pondicherry, Chennai.
        cuisine: Cuisine type to filter by (e.g. 'Biryani', 'Seafood',
                 'South Indian', 'Continental', 'Rajasthani').
        max_price_per_person: Maximum price per person in INR.
        meal_type: Meal service to filter — 'breakfast', 'lunch', 'dinner',
                   or 'snacks'.
        veg_only: True to return only vegetarian-friendly restaurants.
        min_rating: Minimum guest rating (0–5).

    Returns:
        Dict with keys:
          - success (bool)
          - count (int)
          - results (list): each item has id, name, area, cuisine,
            price_per_person, veg_friendly, rating, meal_types,
            opening_hours, distance_from_center_km, tags.
    """
    return _search(
        destination=destination,
        cuisine=cuisine,
        max_price_per_person=max_price_per_person,
        meal_type=meal_type,
        veg_only=veg_only,
        min_rating=min_rating,
    )
