from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from mock_apis.services import search_buses as _search


@tool
def search_buses_tool(
    source: str,
    destination: str,
    date: str,
    bus_type: str | None = None,
    max_price: float | None = None,
) -> dict[str, Any]:
    """Search available buses between two Indian cities.

    Args:
        source: Source city or code — HYD, BLR, MAA, BOM, DEL, GOI, JAI.
        destination: Destination city or code (same set as source).
        date: Travel date in YYYY-MM-DD format.
        bus_type: Bus comfort class — 'sleeper', 'semi_sleeper', 'ac', 'non_ac'.
        max_price: Upper price limit in INR.

    Returns:
        Dict with keys:
          - success (bool)
          - count (int)
          - results (list): each item has id, operator, bus_type, departure_time,
            arrival_time, duration_minutes, price, available_seats, status, rating.
    """
    return _search(
        source=source,
        destination=destination,
        date=date,
        bus_type=bus_type,
        max_price=max_price,
    )
