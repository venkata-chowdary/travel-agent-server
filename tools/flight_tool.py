from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

from langchain_core.tools import tool


def _http_get_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def _as_iso(value: Any) -> str | None:
    if not value:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _normalize_flight(raw: dict[str, Any]) -> dict[str, Any]:
    airline_name = ((raw.get("airline") or {}).get("name")) or (raw.get("airline_name")) or "Unknown"
    flight_iata = ((raw.get("flight") or {}).get("iata")) or ((raw.get("flight") or {}).get("number")) or ""

    dep = raw.get("departure") or {}
    arr = raw.get("arrival") or {}

    departure_time = _as_iso(dep.get("scheduled") or dep.get("estimated") or dep.get("actual"))
    arrival_time = _as_iso(arr.get("scheduled") or arr.get("estimated") or arr.get("actual"))

    status = raw.get("flight_status") or raw.get("status") or "scheduled"

    return {
        "airline": str(airline_name),
        "flight_number": str(flight_iata) if flight_iata else "",
        "departure_time": departure_time,
        "arrival_time": arrival_time,
        "status": str(status),
        # AviationStack doesn't reliably provide stops; omit unless present.
        "stops": raw.get("stops"),
    }


@tool
def search_flights(departure: str, arrival: str) -> list[dict[str, Any]]:
    """
    Search flights between two IATA airport codes using AviationStack.

    Returns structured flight options WITHOUT price. Each option includes:
    - airline
    - flight_number
    - departure_time (ISO string, when available)
    - arrival_time (ISO string, when available)
    - status
    - stops (optional; may be None)
    """

    api_key = os.getenv("AVIATIONSTACK_API_KEY")
    if not api_key:
        raise RuntimeError("AVIATIONSTACK_API_KEY is not set.")

    params = {
        "access_key": api_key,
        "dep_iata": departure.strip().upper(),
        "arr_iata": arrival.strip().upper(),
        "limit": 15,
    }
    url = f"http://api.aviationstack.com/v1/flights?{urlencode(params)}"
    payload = _http_get_json(url)
    data = payload.get("data") or []

    flights: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        normalized = _normalize_flight(item)
        # Keep only minimally valid rows; timestamps may be missing depending on API tier/data.
        if normalized.get("airline") and (normalized.get("departure_time") or normalized.get("arrival_time")):
            flights.append(normalized)

    return flights

