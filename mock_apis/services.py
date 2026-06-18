from __future__ import annotations

import json
import logging
from datetime import date as _Date, timedelta
from pathlib import Path
from typing import Any

from mock_apis.filters import (
    filter_available,
    filter_by_bool,
    filter_by_destination,
    filter_by_field,
    filter_by_max_price,
    filter_by_min_rating,
    filter_by_route,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "mock_data"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load(name: str) -> list[dict[str, Any]]:
    path = DATA_DIR / f"{name}.json"
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("Mock data file missing: %s", path)
        return []
    except json.JSONDecodeError as exc:
        logger.error("Could not parse mock data file %s: %s", path, exc)
        return []


def _inject_date(record: dict[str, Any], date: str) -> dict[str, Any]:
    """Replace stored HH:MM hours with full ISO 8601 timestamps for the given date."""
    r = dict(record)
    r["departure_time"] = f"{date}T{r.pop('departure_hour')}:00"
    offset = int(r.pop("arrival_day_offset", 0))
    if offset:
        arrival_date = (_Date.fromisoformat(date) + timedelta(days=offset)).isoformat()
    else:
        arrival_date = date
    r["arrival_time"] = f"{arrival_date}T{r.pop('arrival_hour')}:00"
    return r


def _err(msg: str) -> dict[str, Any]:
    return {"success": False, "error": msg}


def _route_templates(data: list[dict[str, Any]], source: str, destination: str) -> list[dict[str, Any]]:
    exact = filter_by_route(data, source, destination)
    if exact:
        return exact

    reverse = filter_by_route(data, destination, source)
    if not reverse:
        return []

    src = source.upper().strip()
    dst = destination.upper().strip()
    mirrored = []
    for record in reverse:
        item = dict(record)
        item["id"] = f"{item.get('id', 'route')}_rev"
        item["source"] = src
        item["destination"] = dst
        mirrored.append(item)
    return mirrored


# ---------------------------------------------------------------------------
# Flight search
# ---------------------------------------------------------------------------

def search_flights(
    source: str,
    destination: str,
    date: str,
    objective: str | None = None,
    max_price: float | None = None,
    airline: str | None = None,
) -> dict[str, Any]:
    logger.info("Searching flights: %s → %s on %s (objective=%s, max_price=%s)", source, destination, date, objective, max_price)

    if not source or not destination or not date:
        return _err("Missing required params: source, destination, date")

    data = _load("flights")
    results = _route_templates(data, source, destination)
    results = filter_available(results)

    if airline:
        results = [r for r in results if airline.lower() in r.get("airline", "").lower()]
    if max_price is not None:
        results = filter_by_max_price(results, max_price)

    results = [_inject_date(r, date) for r in results]

    if objective == "cheapest":
        results.sort(key=lambda x: x["price"])
    elif objective == "fastest":
        results.sort(key=lambda x: x["duration_minutes"])
    elif objective == "best_value":
        results.sort(key=lambda x: (-x["rating"], x["price"]))
    else:
        results.sort(key=lambda x: x["price"])

    logger.info("Found %d flight(s)", len(results))
    return {
        "success": True,
        "source": source.upper(),
        "destination": destination.upper(),
        "date": date,
        "count": len(results),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Train search
# ---------------------------------------------------------------------------

def search_trains(
    source: str,
    destination: str,
    date: str,
    class_type: str | None = None,
    max_price: float | None = None,
) -> dict[str, Any]:
    logger.info("Searching trains: %s → %s on %s (class=%s)", source, destination, date, class_type)

    if not source or not destination or not date:
        return _err("Missing required params: source, destination, date")

    data = _load("trains")
    results = _route_templates(data, source, destination)
    results = filter_available(results)

    flattened: list[dict[str, Any]] = []
    for r in results:
        options: list[dict] = r.get("class_options", [])
        if class_type:
            matched = [c for c in options if c["class_type"] == class_type.lower()]
            for c in matched:
                entry = {**r, **c}
                entry.pop("class_options", None)
                flattened.append(entry)
        else:
            if options:
                cheapest = min(options, key=lambda c: c["price"])
                entry = {**r, **cheapest}
                entry.pop("class_options", None)
                flattened.append(entry)

    if max_price is not None:
        flattened = filter_by_max_price(flattened, max_price)

    flattened = [_inject_date(r, date) for r in flattened]
    flattened.sort(key=lambda x: x["price"])

    logger.info("Found %d train option(s)", len(flattened))
    return {
        "success": True,
        "source": source.upper(),
        "destination": destination.upper(),
        "date": date,
        "count": len(flattened),
        "results": flattened,
    }


# ---------------------------------------------------------------------------
# Bus search
# ---------------------------------------------------------------------------

def search_buses(
    source: str,
    destination: str,
    date: str,
    bus_type: str | None = None,
    max_price: float | None = None,
) -> dict[str, Any]:
    logger.info("Searching buses: %s → %s on %s (type=%s)", source, destination, date, bus_type)

    if not source or not destination or not date:
        return _err("Missing required params: source, destination, date")

    data = _load("buses")
    results = _route_templates(data, source, destination)
    results = filter_available(results)

    if bus_type:
        results = filter_by_field(results, "bus_type", bus_type)
    if max_price is not None:
        results = filter_by_max_price(results, max_price)

    results = [_inject_date(r, date) for r in results]
    results.sort(key=lambda x: x["price"])

    logger.info("Found %d bus option(s)", len(results))
    return {
        "success": True,
        "source": source.upper(),
        "destination": destination.upper(),
        "date": date,
        "count": len(results),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Hotel search
# ---------------------------------------------------------------------------

def search_hotels(
    destination: str,
    checkin: str,
    checkout: str,
    guests: int = 1,
    max_price_per_night: float | None = None,
    hotel_type: str | None = None,
    min_rating: float | None = None,
) -> dict[str, Any]:
    logger.info("Searching hotels in %s (%s to %s, %d guest(s))", destination, checkin, checkout, guests)

    if not destination or not checkin or not checkout:
        return _err("Missing required params: destination, checkin, checkout")

    try:
        nights = (_Date.fromisoformat(checkout) - _Date.fromisoformat(checkin)).days
    except ValueError:
        return _err("Invalid date format. Use YYYY-MM-DD.")

    if nights <= 0:
        return _err("checkout must be after checkin")

    data = _load("hotels")
    results = filter_by_destination(data, destination)

    if hotel_type:
        results = filter_by_field(results, "hotel_type", hotel_type)
    if max_price_per_night is not None:
        results = filter_by_max_price(results, max_price_per_night, field="price_per_night")
    if min_rating is not None:
        results = filter_by_min_rating(results, min_rating)

    enriched = []
    for r in results:
        r = dict(r)
        r["total_price"] = round(r["price_per_night"] * nights, 2)
        r["nights"] = nights
        r["checkin"] = checkin
        r["checkout"] = checkout
        enriched.append(r)

    enriched.sort(key=lambda x: x["price_per_night"])

    logger.info("Found %d hotel(s)", len(enriched))
    return {
        "success": True,
        "destination": destination,
        "checkin": checkin,
        "checkout": checkout,
        "nights": nights,
        "count": len(enriched),
        "results": enriched,
    }


# ---------------------------------------------------------------------------
# Restaurant search
# ---------------------------------------------------------------------------

def search_restaurants(
    destination: str,
    cuisine: str | None = None,
    max_price_per_person: float | None = None,
    meal_type: str | None = None,
    veg_only: bool | None = None,
    min_rating: float | None = None,
) -> dict[str, Any]:
    logger.info("Searching restaurants in %s (cuisine=%s, meal=%s)", destination, cuisine, meal_type)

    if not destination:
        return _err("Missing required param: destination")

    data = _load("restaurants")
    results = filter_by_destination(data, destination)

    if cuisine:
        results = [r for r in results if cuisine.lower() in r.get("cuisine", "").lower()]
    if max_price_per_person is not None:
        results = filter_by_max_price(results, max_price_per_person, field="price_per_person")
    if meal_type:
        results = [r for r in results if meal_type.lower() in [m.lower() for m in r.get("meal_types", [])]]
    if veg_only is not None:
        results = filter_by_bool(results, "veg_friendly", veg_only)
    if min_rating is not None:
        results = filter_by_min_rating(results, min_rating)

    results = sorted(results, key=lambda x: -x.get("rating", 0))

    logger.info("Found %d restaurant(s)", len(results))
    return {
        "success": True,
        "destination": destination,
        "count": len(results),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Activity search
# ---------------------------------------------------------------------------

def search_activities(
    destination: str,
    interest: str | None = None,
    max_price: float | None = None,
    duration_hours: float | None = None,
    min_rating: float | None = None,
) -> dict[str, Any]:
    logger.info("Searching activities in %s (interest=%s, max_price=%s)", destination, interest, max_price)

    if not destination:
        return _err("Missing required param: destination")

    data = _load("activities")
    results = filter_by_destination(data, destination)

    if interest:
        results = [
            r for r in results
            if interest.lower() in [t.lower() for t in r.get("tags", [])]
            or interest.lower() == r.get("category", "").lower()
        ]
    if max_price is not None:
        results = filter_by_max_price(results, max_price)
    if duration_hours is not None:
        results = [r for r in results if r.get("duration_hours", 0) <= duration_hours]
    if min_rating is not None:
        results = filter_by_min_rating(results, min_rating)

    results = sorted(results, key=lambda x: -x.get("rating", 0))

    logger.info("Found %d activity/activities", len(results))
    return {
        "success": True,
        "destination": destination,
        "count": len(results),
        "results": results,
    }
