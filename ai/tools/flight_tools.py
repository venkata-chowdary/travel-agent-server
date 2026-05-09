from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from mock_apis.services import search_flights as _search


@tool
def search_flights_tool(
    source: str,
    destination: str,
    date: str,
    objective: str | None = None,
    max_price: float | None = None,
    airline: str | None = None,
) -> dict[str, Any]:
    """Search available flights between two Indian cities.

    Args:
        source: IATA departure code — HYD, BLR, MAA, BOM, DEL, GOI, JAI, COK.
        destination: IATA arrival code (same set as source).
        date: Travel date in YYYY-MM-DD format.
        objective: Sort strategy — 'cheapest', 'fastest', or 'best_value'.
        max_price: Upper price limit in INR (e.g. 5000).
        airline: Filter by airline name (e.g. 'IndiGo', 'Air India', 'Vistara').

    Returns:
        Dict with keys:
          - success (bool)
          - count (int): number of matching flights
          - results (list): each item has id, airline, flight_number, departure_time,
            arrival_time, duration_minutes, price, status, available_seats,
            baggage_kg, refundable, rating.
    """
    return _search(
        source=source,
        destination=destination,
        date=date,
        objective=objective,
        max_price=max_price,
        airline=airline,
    )
