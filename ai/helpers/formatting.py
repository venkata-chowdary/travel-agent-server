from __future__ import annotations

from ai.schemas.experience import ExperienceContext
from ai.schemas import PreferenceContext, TravelPreferences
from ai.schemas.hotel import HotelOption
from ai.schemas.transport import TransportOption
from ai.schemas.weather import WeatherForecastResponse


def format_preferences_block(
    prefs: PreferenceContext | TravelPreferences | None,
) -> str:
    if prefs is None:
        return ""
    if isinstance(prefs, PreferenceContext):
        return _format_preference_context(prefs)
    return _format_travel_preferences(prefs)


def _format_preference_context(ctx: PreferenceContext) -> str:
    fields = {
        "Travel style": ctx.travel_style,
        "Budget style": ctx.budget_style,
        "Preferred transport": ", ".join(ctx.preferred_transport) if ctx.preferred_transport else None,
        "Food preference": ctx.food_preference,
        "Hotel preference": ctx.hotel_preference,
        "Avoid": ", ".join(ctx.avoid) if ctx.avoid else None,
        "Origin": ctx.origin,
        "Preferred currency": ctx.currency,
        "Memory confidence": f"{ctx.memory_confidence:.0%}" if ctx.memory_confidence else None,
    }
    set_fields = {k: v for k, v in fields.items() if v is not None}
    if not set_fields:
        return ""
    lines = "\n".join(f"  - {k}: {v}" for k, v in set_fields.items())
    return f"\n\nTraveler profile (synthesized from preferences and past trips):\n{lines}"


def _format_travel_preferences(prefs: TravelPreferences) -> str:
    fields = {
        "Budget range": prefs.budget_range,
        "Travel style": ", ".join(prefs.travel_style) if prefs.travel_style else None,
        "Dietary restrictions": ", ".join(prefs.dietary_restrictions) if prefs.dietary_restrictions else None,
        "Accommodation": prefs.accommodation_type,
        "Trip pace": prefs.pace,
        "Origin": prefs.origin,
        "Preferred currency": prefs.currency,
    }
    set_fields = {k: v for k, v in fields.items() if v is not None}
    if not set_fields:
        return ""
    lines = "\n".join(f"  - {k}: {v}" for k, v in set_fields.items())
    return f"\n\nTraveler preferences (use these to personalize your response):\n{lines}"


def format_weather_block(forecast: WeatherForecastResponse | None) -> str:
    if forecast is None or not forecast.daily_forecast:
        return ""  # no real data — main LLM uses general knowledge
    lines = [f"\n\n[WEATHER — from WeatherAgent, authoritative. Use exactly as provided:]"]
    lines.append(f"  Destination: {forecast.destination}")
    lines.append(f"  Summary: {forecast.summary}")
    if forecast.requires_replanning:
        lines.append("  WARNING: Severe weather — recommend replanning or extra preparation.")
    for day in forecast.daily_forecast:
        lines.append(
            f"  {day.date}: {day.condition}, {day.temperature}, "
            f"rain {day.rain_probability}%, risk={day.risk_level}"
        )
    if forecast.trip_risks:
        lines.append("  Risks:")
        for risk in forecast.trip_risks:
            lines.append(
                f"    Day {risk.day} [{risk.severity}] {risk.risk_type}: {risk.recommendation}"
            )
    return "\n".join(lines)


def format_experience_block(context: ExperienceContext | None, num_travelers: int = 1) -> str:
    if context is None or (not context.activities and not context.restaurants):
        return ""

    lines = ["\n\n[REQUIRED RECOMMENDED EXPERIENCE CONTEXT - from ExperienceAgent, authoritative. Use provided names/prices:]"]
    lines.append(f"  Destination: {context.destination}")
    if context.activities:
        lines.append("  Recommended activities:")
        for activity in context.activities:
            weather_note = " weather-sensitive" if activity.weather_sensitive else ""
            location = f", {activity.location}" if activity.location else ""
            best_time = f", best time: {activity.best_time}" if activity.best_time else ""
            lines.append(
                f"    - {activity.name} ({activity.category}{location}): "
                f"INR {activity.price} per person, {activity.duration_hours:g}h, "
                f"rating {activity.rating}/5{best_time}{weather_note}"
            )
    if context.restaurants:
        lines.append("  Recommended restaurants:")
        for restaurant in context.restaurants:
            area = f", {restaurant.area}" if restaurant.area else ""
            meals = f", meals: {', '.join(restaurant.meal_types)}" if restaurant.meal_types else ""
            veg = ", veg-friendly" if restaurant.veg_friendly else ""
            lines.append(
                f"    - {restaurant.name} ({restaurant.cuisine}{area}): "
                f"INR {restaurant.price_per_person} per person, rating {restaurant.rating}/5"
                f"{meals}{veg}"
            )
    lines.append(f"  Travelers: {num_travelers}")
    lines.append("  The server will set budget.activities and budget.food from these recommended prices.")
    return "\n".join(lines)


def format_hotel_block(option: HotelOption | None) -> str:
    if not option:
        return ""
    lines = ["\n\n[REQUIRED SELECTED HOTEL - chosen by the user. Treat as authoritative:]"]
    lines.append(f"  {option.name} ({option.hotel_type}){', ' + option.area if option.area else ''}")
    lines.append(
        f"  Rating: {option.rating}/5, INR {option.price_per_night:,}/night, "
        f"Total: INR {option.total_price:,} ({option.total_price // option.price_per_night if option.price_per_night else '?'} nights)"
    )
    if option.breakfast_included:
        lines.append("  Breakfast included")
    if option.refundable:
        lines.append("  Refundable")
    lines.append(f"  Set budget.stay to exactly INR {option.total_price}")
    return "\n".join(lines)


def format_transport_block(options: list[TransportOption] | None, num_travelers: int = 1) -> str:
    if not options:
        return ""

    lines = ["\n\n[REQUIRED SELECTED TRANSPORT - chosen by the user. Treat as authoritative:]"]
    per_person_total = 0
    for option in options:
        per_person_total += option.price
        details = []
        if option.details.get("flight_number"):
            details.append(f"flight {option.details['flight_number']}")
        if option.details.get("train_number"):
            details.append(f"train {option.details['train_number']}")
        if option.details.get("class_type"):
            details.append(f"class {option.details['class_type']}")
        if option.details.get("bus_type"):
            details.append(f"{option.details['bus_type']} bus")
        detail_text = f" ({', '.join(details)})" if details else ""
        lines.append(
            f"  {option.leg}: {option.mode} via {option.provider}{detail_text}, "
            f"{option.from_}->{option.to}, {option.depart}-{option.arrive}, "
            f"{option.duration}, INR {option.price} per person"
        )
    if num_travelers > 1:
        lines.append(f"  Travelers: {num_travelers}")
        lines.append(f"  Per-person transport total: INR {per_person_total}")
        lines.append(f"  Total transport cost ({num_travelers} travelers): INR {per_person_total * num_travelers}")
        lines.append(f"  Set budget.flights to exactly INR {per_person_total * num_travelers}")
    else:
        lines.append(f"  Total selected transport cost: INR {per_person_total}")
        lines.append(f"  Set budget.flights to exactly INR {per_person_total}")
    return "\n".join(lines)
