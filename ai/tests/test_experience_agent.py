from __future__ import annotations

from ai.agents.experience_agent import build_experience_context
from ai.agents.planner import _experience_costs
from ai.helpers import format_experience_block
from ai.schemas import PreferenceContext
from ai.schemas.weather import DailyForecast, WeatherForecastResponse


def _rainy_forecast() -> WeatherForecastResponse:
    return WeatherForecastResponse(
        destination="Goa",
        summary="Rain risk across the trip.",
        daily_forecast=[
            DailyForecast(
                day=1,
                date="2026-07-01",
                condition="rain",
                temperature="24C - 30C",
                rain_probability=80,
                risk_level="high",
            )
        ],
        trip_risks=[],
        requires_replanning=True,
    )


def test_build_experience_context_finds_supported_destination_options() -> None:
    context = build_experience_context(destination="Goa", days=3, travelers=2)

    assert context.destination == "Goa"
    assert context.activities
    assert context.restaurants
    assert len(context.activities) <= 6
    assert len(context.restaurants) <= 6
    assert context.supervisor_note


def test_build_experience_context_applies_vegetarian_preference() -> None:
    prefs = PreferenceContext(food_preference="veg")

    context = build_experience_context(
        destination="Goa",
        days=2,
        travelers=1,
        preferences=prefs,
    )

    assert context.restaurants
    assert all(restaurant.veg_friendly for restaurant in context.restaurants)


def test_build_experience_context_broadens_activity_filters(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_search_activities(**kwargs):
        calls.append(kwargs)
        if kwargs.get("interest"):
            return {"success": True, "count": 0, "results": []}
        return {
            "success": True,
            "count": 1,
            "results": [
                {
                    "id": "act_fallback",
                    "name": "Fallback Heritage Walk",
                    "provider": "MockXplore",
                    "category": "culture",
                    "price": 500,
                    "duration_hours": 2,
                    "rating": 4.4,
                    "location": "Old Town",
                    "difficulty": "easy",
                    "best_time": "Morning",
                    "weather_sensitive": False,
                    "tags": ["culture"],
                }
            ],
        }

    monkeypatch.setattr("ai.agents.experience_agent.search_activities", fake_search_activities)
    prefs = PreferenceContext(travel_style="adventure")

    context = build_experience_context(
        destination="Goa",
        days=2,
        travelers=1,
        preferences=prefs,
    )

    assert len(calls) >= 2
    assert calls[0]["interest"] == "adventure"
    assert calls[1].get("interest") is None
    assert [activity.id for activity in context.activities] == ["act_fallback"]


def test_build_experience_context_penalizes_weather_sensitive_activities() -> None:
    context = build_experience_context(
        destination="Goa",
        days=3,
        travelers=1,
        weather=_rainy_forecast(),
    )

    assert context.activities
    assert context.activities[0].weather_sensitive is False


def test_build_experience_context_returns_empty_for_unsupported_destination() -> None:
    context = build_experience_context(destination="Atlantis", days=2, travelers=1)

    assert context.activities == []
    assert context.restaurants == []
    assert "No mock activities" in context.summary


def test_format_experience_block_includes_authoritative_details() -> None:
    context = build_experience_context(destination="Goa", days=2, travelers=2)

    block = format_experience_block(context, num_travelers=2)

    assert "EXPERIENCE CONTEXT" in block
    assert context.activities[0].name in block
    assert context.restaurants[0].name in block
    assert "INR" in block
    assert "authoritative" in block


def test_experience_costs_scale_recommended_items_by_travelers() -> None:
    context = build_experience_context(destination="Goa", days=2, travelers=3)

    activities_total, food_total = _experience_costs(context, num_travelers=3)

    assert activities_total == sum(activity.price for activity in context.activities) * 3
    assert food_total == sum(restaurant.price_per_person for restaurant in context.restaurants) * 3
