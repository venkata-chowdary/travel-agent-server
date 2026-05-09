from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from mock_apis.services import search_trains as _search


@tool
def search_trains_tool(
    source: str,
    destination: str,
    date: str,
    class_type: str | None = None,
    max_price: float | None = None,
) -> dict[str, Any]:
    """Search available trains between two Indian cities.

    Args:
        source: Source station code — HYD, BLR, MAA, BOM, DEL, GOI, JAI.
        destination: Destination station code (same set as source).
        date: Travel date in YYYY-MM-DD format.
        class_type: Coach class — 'sleeper', '3ac', '2ac', 'chair_car'.
                    If omitted, the cheapest available class is returned per train.
        max_price: Upper price limit in INR per person.

    Returns:
        Dict with keys:
          - success (bool)
          - count (int)
          - results (list): each item has id, train_name, train_number, departure_time,
            arrival_time, duration_minutes, class_type, price, available_seats,
            status, rating.
    """
    return _search(
        source=source,
        destination=destination,
        date=date,
        class_type=class_type,
        max_price=max_price,
    )
