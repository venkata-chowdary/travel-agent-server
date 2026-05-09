from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from mock_apis.services import search_hotels as _search


@tool
def search_hotels_tool(
    destination: str,
    checkin: str,
    checkout: str,
    guests: int = 1,
    max_price_per_night: float | None = None,
    hotel_type: str | None = None,
    min_rating: float | None = None,
) -> dict[str, Any]:
    """Search available hotels at a destination.

    Args:
        destination: City name — Goa, Bengaluru, Hyderabad, Mumbai, Delhi,
                     Jaipur, Kochi, Manali, Pondicherry, Chennai.
        checkin: Check-in date in YYYY-MM-DD format.
        checkout: Check-out date in YYYY-MM-DD format.
        guests: Number of guests (default 1). Does not filter results but is
                recorded for context.
        max_price_per_night: Upper limit on nightly room rate in INR.
        hotel_type: Category — 'budget', 'mid_range', or 'luxury'.
        min_rating: Minimum acceptable guest rating (0–5).

    Returns:
        Dict with keys:
          - success (bool)
          - nights (int): number of stay nights computed from checkin/checkout
          - count (int)
          - results (list): each item has id, name, area, hotel_type,
            price_per_night, total_price, rating, amenities,
            distance_from_center_km, available_rooms, refundable,
            breakfast_included.
    """
    return _search(
        destination=destination,
        checkin=checkin,
        checkout=checkout,
        guests=guests,
        max_price_per_night=max_price_per_night,
        hotel_type=hotel_type,
        min_rating=min_rating,
    )
