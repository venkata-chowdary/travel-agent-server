from __future__ import annotations

import logging
from typing import Any

from ai.schemas import PreferenceContext
from ai.schemas.experience import ActivityOption, ExperienceContext, RestaurantOption
from ai.schemas.weather import WeatherForecastResponse
from ai.state import TravelState, set_status
from mock_apis.services import search_activities, search_restaurants

logger = logging.getLogger(__name__)

_STYLE_TO_INTEREST = {
    "adventure": "adventure",
    "adventurous": "adventure",
    "relaxation": "wellness",
    "relaxed": "wellness",
    "cultural": "culture",
    "culture": "culture",
    "foodie": "food",
    "food": "food",
    "family": "culture",
    "solo": "culture",
    "nightlife": "nightlife",
    "nature": "nature",
    "beach": "beach",
}


def _cap_for_days(days: int) -> int:
    return min(max(days * 2, 4), 10)


def _activity_max_price(budget_style: str | None) -> float | None:
    if budget_style == "budget":
        return 1500.0
    if budget_style == "mid-range":
        return 3500.0
    return None


def _restaurant_max_price(budget_style: str | None) -> float | None:
    if budget_style == "budget":
        return 700.0
    if budget_style == "mid-range":
        return 1500.0
    return None


def _interest_from_preferences(preferences: PreferenceContext | None) -> str | None:
    if not preferences or not preferences.travel_style:
        return None
    text = preferences.travel_style.lower()
    for key, interest in _STYLE_TO_INTEREST.items():
        if key in text:
            return interest
    return None


def _veg_only(preferences: PreferenceContext | None) -> bool | None:
    if not preferences or not preferences.food_preference:
        return None
    food = preferences.food_preference.lower()
    if any(token in food for token in ("veg", "vegetarian", "vegan", "jain")):
        return True
    return None


def _has_rain_risk(forecast: WeatherForecastResponse | None) -> bool:
    if not forecast:
        return False
    return any(day.risk_level in {"medium", "high"} for day in forecast.daily_forecast or [])


def _normalise_activity(item: dict[str, Any]) -> ActivityOption:
    return ActivityOption(
        id=str(item["id"]),
        name=str(item["name"]),
        provider=str(item.get("provider") or "MockXplore"),
        category=str(item.get("category") or "activity"),
        price=int(item.get("price") or 0),
        duration_hours=float(item.get("duration_hours") or 0),
        rating=float(item.get("rating") or 0),
        location=item.get("location"),
        difficulty=item.get("difficulty"),
        best_time=item.get("best_time"),
        weather_sensitive=bool(item.get("weather_sensitive", False)),
        tags=list(item.get("tags") or []),
    )


def _normalise_restaurant(item: dict[str, Any]) -> RestaurantOption:
    return RestaurantOption(
        id=str(item["id"]),
        name=str(item["name"]),
        provider=str(item.get("provider") or "MockDine"),
        area=item.get("area"),
        cuisine=str(item.get("cuisine") or "Local"),
        price_per_person=int(item.get("price_per_person") or 0),
        veg_friendly=bool(item.get("veg_friendly", False)),
        rating=float(item.get("rating") or 0),
        meal_types=list(item.get("meal_types") or []),
        opening_hours=item.get("opening_hours"),
        distance_from_center_km=item.get("distance_from_center_km"),
        tags=list(item.get("tags") or []),
    )


def _activity_rank(option: ActivityOption, rain_risk: bool) -> tuple[int, float, int, float]:
    weather_penalty = 1 if rain_risk and option.weather_sensitive else 0
    return (weather_penalty, -option.rating, option.price, option.duration_hours)


def _restaurant_rank(option: RestaurantOption) -> tuple[float, int, float]:
    distance = option.distance_from_center_km if option.distance_from_center_km is not None else 999.0
    return (-option.rating, option.price_per_person, distance)


def _search_activity_options(
    *,
    destination: str,
    cap: int,
    preferences: PreferenceContext | None,
    weather: WeatherForecastResponse | None,
) -> list[ActivityOption]:
    budget_style = preferences.budget_style if preferences else None
    result = search_activities(
        destination=destination,
        interest=_interest_from_preferences(preferences),
        max_price=_activity_max_price(budget_style),
        min_rating=4.0,
    )
    if not result.get("success") or not result.get("results"):
        result = search_activities(destination=destination, min_rating=4.0)
    if not result.get("success") or not result.get("results"):
        result = search_activities(destination=destination)

    rain_risk = _has_rain_risk(weather)
    options = [_normalise_activity(item) for item in result.get("results") or []]
    options.sort(key=lambda option: _activity_rank(option, rain_risk))
    return options[:cap]


def _search_restaurant_options(
    *,
    destination: str,
    cap: int,
    preferences: PreferenceContext | None,
) -> list[RestaurantOption]:
    budget_style = preferences.budget_style if preferences else None
    result = search_restaurants(
        destination=destination,
        max_price_per_person=_restaurant_max_price(budget_style),
        veg_only=_veg_only(preferences),
        min_rating=4.0,
    )
    if not result.get("success") or not result.get("results"):
        result = search_restaurants(
            destination=destination,
            veg_only=_veg_only(preferences),
            min_rating=4.0,
        )
    if not result.get("success") or not result.get("results"):
        result = search_restaurants(destination=destination)

    options = [_normalise_restaurant(item) for item in result.get("results") or []]
    options.sort(key=_restaurant_rank)
    return options[:cap]


def build_experience_context(
    *,
    destination: str,
    days: int,
    travelers: int,
    preferences: PreferenceContext | None = None,
    weather: WeatherForecastResponse | None = None,
) -> ExperienceContext:
    cap = _cap_for_days(days)
    activities = _search_activity_options(
        destination=destination,
        cap=cap,
        preferences=preferences,
        weather=weather,
    )
    restaurants = _search_restaurant_options(
        destination=destination,
        cap=cap,
        preferences=preferences,
    )

    if not activities and not restaurants:
        return ExperienceContext(
            destination=destination,
            days=days,
            travelers=travelers,
            activities=[],
            restaurants=[],
            summary=f"No mock activities or restaurants were found for {destination}.",
            supervisor_note=f"No experience options found for {destination}; planner may use general knowledge.",
        )

    summary = (
        f"Found {len(activities)} activity option(s) and {len(restaurants)} restaurant option(s) "
        f"for {destination}."
    )
    recommended_bits = []
    if activities:
        recommended_bits.append(f"top activity: {activities[0].name}")
    if restaurants:
        recommended_bits.append(f"top restaurant: {restaurants[0].name}")
    supervisor_note = summary
    if recommended_bits:
        supervisor_note += " " + "; ".join(recommended_bits) + "."

    return ExperienceContext(
        destination=destination,
        days=days,
        travelers=travelers,
        activities=activities,
        restaurants=restaurants,
        summary=summary,
        supervisor_note=supervisor_note,
    )


def experience_agent_node(state: TravelState) -> dict:
    destination = state.get("destination")
    days = state.get("trip_duration_days") or 3
    travelers = state.get("num_travelers") or 1
    if not destination:
        logger.info("ExperienceAgent skipped - missing destination")
        return {"workflow_statuses": set_status(state, "experience", "failed")}

    logger.info("ExperienceAgent running - %s, %s day(s)", destination, days)
    context = build_experience_context(
        destination=destination,
        days=days,
        travelers=travelers,
        preferences=state.get("preference_context"),
        weather=state.get("weather_forecast"),
    )
    status = "succeeded" if (context.activities or context.restaurants) else "empty"
    logger.info(
        "ExperienceAgent done - %d activities, %d restaurants, status=%s",
        len(context.activities), len(context.restaurants), status,
    )
    return {
        "experience_context": context,
        "workflow_statuses": set_status(state, "experience", status),
    }
