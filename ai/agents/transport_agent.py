from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

from ai.schemas import PreferenceContext
from ai.schemas.transport import TransportChoiceResponse, TransportLeg, TransportOption
from ai.state import TravelState, get_origin, get_trip_dates, has_transport_options, set_status
from mock_apis.services import search_buses, search_flights, search_trains

logger = logging.getLogger(__name__)


CITY_CODES = {
    "agra": "AGR",
    "amritsar": "ATQ",
    "andaman": "IXZ",
    "bangalore": "BLR",
    "bengaluru": "BLR",
    "blr": "BLR",
    "bom": "BOM",
    "chennai": "MAA",
    "cochin": "COK",
    "cok": "COK",
    "del": "DEL",
    "delhi": "DEL",
    "dehradun": "DED",
    "goa": "GOI",
    "goi": "GOI",
    "hyderabad": "HYD",
    "hyd": "HYD",
    "jaipur": "JAI",
    "jai": "JAI",
    "jodhpur": "JDH",
    "kolkata": "CCU",
    "maa": "MAA",
    "mumbai": "BOM",
    "mysore": "MYQ",
    "rishikesh": "RSH",
    "shimla": "SML",
    "udaipur": "UDR",
    "varanasi": "VNS",
}


def resolve_city_code(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip().lower()
    if cleaned in CITY_CODES:
        return CITY_CODES[cleaned]
    if len(cleaned) == 3 and cleaned.isalpha():
        return cleaned.upper()
    return None


def build_transport_choice(
    *,
    origin: str,
    destination: str,
    start_date: str,
    days: int,
    travelers: int,
    preferences: PreferenceContext | None = None,
) -> TransportChoiceResponse:
    source_code = resolve_city_code(origin)
    destination_code = resolve_city_code(destination)
    if not source_code or not destination_code:
        missing = origin if not source_code else destination
        return TransportChoiceResponse(
            origin=origin,
            destination=destination,
            start_date=start_date,
            end_date=_end_date(start_date, days),
            days=days,
            travelers=travelers,
            unavailable_modes=[f"Unsupported city: {missing}"],
            summary=(
                "I couldn't match one of the cities to the mock transport network. "
                "Try a supported Indian city such as Hyderabad, Goa, Bangalore, Mumbai, Delhi, Jaipur, or Chennai."
            ),
        )

    end_date = _end_date(start_date, days)
    preferred = preferences.preferred_transport if preferences else []
    outbound = _search_leg(
        leg="outbound",
        source=source_code,
        destination=destination_code,
        travel_date=start_date,
        preferred_transport=preferred,
    )
    return_options = _search_leg(
        leg="return",
        source=destination_code,
        destination=source_code,
        travel_date=end_date,
        preferred_transport=preferred,
    )

    unavailable = []
    for leg_name, options in (("outbound", outbound), ("return", return_options)):
        available_modes = {option.mode for option in options}
        for mode in ("flight", "train", "bus"):
            if mode not in available_modes:
                unavailable.append(f"{leg_name} {mode}")

    recommended_outbound = _recommended(outbound, preferred)
    recommended_return = _recommended(return_options, preferred)
    total_options = len(outbound) + len(return_options)
    if total_options == 0:
        summary = (
            f"I couldn't find mock transport options for {origin} to {destination} "
            f"for {start_date}. I can check again, use different route or date details, "
            "or continue without a selected transport option."
        )
    else:
        summary = (
            f"I found {total_options} transport option{'s' if total_options != 1 else ''} "
            f"for {origin} to {destination}. Pick the outbound"
        )
        if return_options:
            summary += " and return"
        summary += " option you want me to plan around."

    return TransportChoiceResponse(
        origin=origin,
        destination=destination,
        start_date=start_date,
        end_date=end_date,
        days=days,
        travelers=travelers,
        outbound_options=outbound,
        return_options=return_options,
        recommended_outbound_id=recommended_outbound.id if recommended_outbound else None,
        recommended_return_id=recommended_return.id if recommended_return else None,
        unavailable_modes=unavailable,
        summary=summary,
    )


def _end_date(start_date: str, days: int) -> str:
    try:
        start = date.fromisoformat(start_date)
    except ValueError:
        start = date.today()
    return (start + timedelta(days=max(days - 1, 0))).isoformat()


def _search_leg(
    *,
    leg: TransportLeg,
    source: str,
    destination: str,
    travel_date: str,
    preferred_transport: list[str],
) -> list[TransportOption]:
    raw_results = [
        ("flight", search_flights(source, destination, travel_date, objective="best_value")),
        ("train", search_trains(source, destination, travel_date)),
        ("bus", search_buses(source, destination, travel_date)),
    ]
    options: list[TransportOption] = []
    for mode, payload in raw_results:
        if not payload.get("success"):
            logger.info("Transport %s search failed: %s", mode, payload.get("error"))
            continue
        mode_options = [_normalise_option(mode, leg, source, destination, item) for item in payload.get("results", [])]
        mode_options.sort(key=lambda option: _rank_key(option, preferred_transport))
        options.extend(mode_options[:3])
    options.sort(key=lambda option: _rank_key(option, preferred_transport))
    return options


def _normalise_option(
    mode: str,
    leg: TransportLeg,
    source: str,
    destination: str,
    item: dict[str, Any],
) -> TransportOption:
    depart = _time_from_iso(item["departure_time"])
    arrive = _time_from_iso(item["arrival_time"])
    provider = (
        item.get("airline")
        or item.get("train_name")
        or item.get("operator")
        or item.get("provider")
        or mode.title()
    )
    details: dict[str, Any] = {}
    if mode == "flight":
        details = {
            "flight_number": item.get("flight_number"),
            "baggage_kg": item.get("baggage_kg"),
            "refundable": item.get("refundable"),
            "status": item.get("status"),
        }
    elif mode == "train":
        details = {
            "train_number": item.get("train_number"),
            "class_type": item.get("class_type"),
            "status": item.get("status"),
        }
    else:
        details = {
            "bus_type": item.get("bus_type"),
            "status": item.get("status"),
        }

    return TransportOption.model_validate({
        "id": f"{leg}_{item['id']}",
        "mode": mode,
        "leg": leg,
        "provider": str(provider),
        "from": source,
        "to": destination,
        "depart": depart,
        "arrive": arrive,
        "duration": _duration(item.get("duration_minutes", 0)),
        "price": int(item.get("price") or 0),
        "available_seats": int(item.get("available_seats") or 0),
        "rating": item.get("rating"),
        "details": {k: v for k, v in details.items() if v is not None},
    })


def _time_from_iso(value: str) -> str:
    return value.split("T", 1)[1][:5] if "T" in value else value[:5]


def _duration(minutes: int) -> str:
    hours, mins = divmod(int(minutes), 60)
    if hours and mins:
        return f"{hours}h {mins}m"
    if hours:
        return f"{hours}h"
    return f"{mins}m"


def _recommended(
    options: list[TransportOption],
    preferred_transport: list[str],
) -> TransportOption | None:
    if not options:
        return None
    return min(options, key=lambda option: _rank_key(option, preferred_transport))


def _rank_key(option: TransportOption, preferred_transport: list[str]) -> tuple[int, int, float, int]:
    preference_text = " ".join(preferred_transport).lower()
    preference_penalty = 0 if option.mode in preference_text else 1
    rating_penalty = -(option.rating or 0)
    return (preference_penalty, option.price, rating_penalty, option.available_seats * -1)


def _build_supervisor_note(choice: TransportChoiceResponse) -> str:
    if not has_transport_options(choice):
        return (
            f"No transport options found for {choice.origin} → {choice.destination} on {choice.start_date}. "
            "Different dates or a nearby departure city might help."
        )
    outbound_modes = {o.mode for o in choice.outbound_options}
    return_modes = {o.mode for o in choice.return_options}
    all_modes = outbound_modes | return_modes
    missing = [m for m in ("flight", "train", "bus") if m not in all_modes]
    n_total = len(choice.outbound_options) + len(choice.return_options)
    note = f"Found {n_total} option(s) across {', '.join(sorted(all_modes))} for {choice.origin} → {choice.destination}."
    if missing:
        note += f" {', '.join(m.title() for m in missing)} not available on this route."
    return note


async def transport_agent_node(state: TravelState) -> dict:
    origin = get_origin(state)
    destination = state.get("destination")
    trip_dates = get_trip_dates(state)
    if not origin or not destination:
        logger.info("TransportAgent skipped — missing origin or destination")
        return {"workflow_statuses": set_status(state, "transport", "failed")}

    logger.info("TransportAgent running — %s to %s on %s", origin, destination, trip_dates[0])
    choice = build_transport_choice(
        origin=origin,
        destination=destination,
        start_date=trip_dates[0],
        days=state.get("trip_duration_days") or len(trip_dates),
        travelers=1,
        preferences=state.get("preference_context"),
    )
    choice = choice.model_copy(update={"supervisor_note": _build_supervisor_note(choice)})
    logger.info(
        "TransportAgent found %s outbound and %s return option(s)",
        len(choice.outbound_options), len(choice.return_options),
    )
    return {
        "transport_choice": choice,
        "workflow_statuses": set_status(
            state, "transport", "waiting_for_user" if has_transport_options(choice) else "empty"
        ),
    }
