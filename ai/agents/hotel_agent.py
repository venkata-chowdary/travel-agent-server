from __future__ import annotations

import logging
from datetime import date, timedelta

from ai.schemas import PreferenceContext
from ai.schemas.hotel import HotelChoiceResponse, HotelOption
from ai.state import TravelState, set_status
from mock_apis.services import search_hotels

logger = logging.getLogger(__name__)

_BUDGET_TYPES = {"budget", "hostel"}
_LUXURY_TYPES = {"luxury", "boutique"}


def _score_option(option: HotelOption, budget_style: str | None) -> tuple:
    """Lower score = better rank."""
    type_match = 0
    if budget_style == "budget" and option.hotel_type in _BUDGET_TYPES:
        type_match = -1
    elif budget_style == "luxury" and option.hotel_type in _LUXURY_TYPES:
        type_match = -1
    elif budget_style == "mid-range" and option.hotel_type not in _BUDGET_TYPES | _LUXURY_TYPES:
        type_match = -1
    return (type_match, -option.rating, option.price_per_night)


def build_hotel_choice(
    *,
    destination: str,
    checkin: str,
    checkout: str,
    nights: int,
    travelers: int,
    preferences: PreferenceContext | None = None,
) -> HotelChoiceResponse:
    budget_style = preferences.budget_style if preferences else None
    hotel_pref = preferences.hotel_preference if preferences else None

    max_price: float | None = None
    hotel_type_filter: str | None = None
    min_rating: float | None = None

    if budget_style == "budget":
        max_price = 4000.0
    elif budget_style == "luxury":
        min_rating = 4.0

    if hotel_pref and hotel_pref.lower() in {"hotel", "hostel", "resort", "boutique", "airbnb"}:
        hotel_type_filter = hotel_pref.lower()

    result = search_hotels(
        destination=destination,
        checkin=checkin,
        checkout=checkout,
        guests=travelers,
        max_price_per_night=max_price,
        hotel_type=hotel_type_filter,
        min_rating=min_rating,
    )

    if not result.get("success") or not result.get("results"):
        # Retry without filters if nothing matched
        result = search_hotels(destination=destination, checkin=checkin, checkout=checkout, guests=travelers)

    raw = result.get("results") or []

    options: list[HotelOption] = []
    for r in raw:
        price_per_night = int(r.get("price_per_night", 0))
        total = price_per_night * nights
        options.append(HotelOption(
            id=r["id"],
            name=r["name"],
            provider=r.get("provider", "MockStay"),
            area=r.get("area"),
            hotel_type=r.get("hotel_type", "hotel"),
            price_per_night=price_per_night,
            total_price=total,
            rating=float(r.get("rating", 0.0)),
            amenities=r.get("amenities") or [],
            distance_from_center_km=r.get("distance_from_center_km"),
            available_rooms=int(r.get("available_rooms", 0)),
            refundable=bool(r.get("refundable", False)),
            breakfast_included=bool(r.get("breakfast_included", False)),
        ))

    options.sort(key=lambda o: _score_option(o, budget_style))
    options = options[:5]

    if not options:
        return HotelChoiceResponse(
            destination=destination,
            checkin=checkin,
            checkout=checkout,
            nights=nights,
            travelers=travelers,
            options=[],
            summary=(
                f"I couldn't find any hotels in {destination} for those dates. "
                "You may want to search manually or adjust your dates."
            ),
            supervisor_note=f"No hotels found for {destination} ({checkin} to {checkout}).",
        )

    recommended = options[0]
    style_note = f" Based on your {budget_style} budget preference." if budget_style else ""
    summary = (
        f"Here are {len(options)} hotel option{'s' if len(options) != 1 else ''} in {destination} "
        f"for {nights} night{'s' if nights != 1 else ''} ({checkin} to {checkout}).{style_note} "
        f"I recommend {recommended.name} — rated {recommended.rating}/5, "
        f"₹{recommended.price_per_night:,}/night (₹{recommended.total_price:,} total). "
        "Pick the one that works best for you."
    )
    supervisor_note = f"{len(options)} hotels found; recommended {recommended.name} ({recommended.hotel_type}, {recommended.rating}/5)."

    return HotelChoiceResponse(
        destination=destination,
        checkin=checkin,
        checkout=checkout,
        nights=nights,
        travelers=travelers,
        options=options,
        recommended_id=recommended.id,
        summary=summary,
        supervisor_note=supervisor_note,
    )


def _checkout_date(checkin: str, days: int) -> str:
    try:
        start = date.fromisoformat(checkin)
    except ValueError:
        start = date.today()
    return (start + timedelta(days=days)).isoformat()


def hotel_agent_node(state: TravelState) -> dict:
    logger.info("Hotel agent searching options...")

    destination = state.get("destination") or ""
    checkin = state.get("trip_start_date") or date.today().isoformat()
    days = state.get("trip_duration_days") or 3
    checkout = _checkout_date(checkin, days)
    prefs = state.get("preference_context")
    travelers = 1

    result = build_hotel_choice(
        destination=destination,
        checkin=checkin,
        checkout=checkout,
        nights=days,
        travelers=travelers,
        preferences=prefs,
    )

    status = "waiting_for_user" if result.options else "empty"
    logger.info("Hotel agent done — %d option(s), status=%s", len(result.options), status)

    return {
        "hotel_choice": result,
        "workflow_statuses": set_status(state, "hotel", status),
    }
