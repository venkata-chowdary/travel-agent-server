import os

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/test")
os.environ.setdefault("JWT_SECRET_KEY", "x" * 32)

from ai.agent import _clarification_response
from ai.schemas import PreferenceContext


def base_state(message: str, **overrides):
    state = {
        "user_id": "user-1",
        "user_message": message,
        "messages": [],
        "next": "",
        "origin": None,
        "destination": None,
        "trip_duration_days": None,
        "clarification_checked": False,
        "clarification_response": None,
        "preference_context": PreferenceContext(),
        "weather_forecast": None,
        "transport_choice": None,
        "selected_transport_options": None,
        "structured_response": None,
    }
    state.update(overrides)
    return state


def test_clarifier_asks_for_core_missing_details():
    response = _clarification_response(base_state("Plan a trip"))

    assert response is not None
    assert response.response_type == "clarification"
    assert len(response.questions) == 2  # destination + duration; budget is no longer required


def test_clarifier_uses_saved_budget_preference():
    response = _clarification_response(base_state(
        "Plan a 3 day Goa trip",
        destination="Goa",
        trip_duration_days=3,
        preference_context=PreferenceContext(budget_style="budget", currency="INR", home_city="Hyderabad"),
    ))

    assert response is None


def test_clarifier_asks_for_origin_when_missing_from_profile():
    response = _clarification_response(base_state(
        "Plan a 3 day Goa trip",
        destination="Goa",
        trip_duration_days=3,
        preference_context=PreferenceContext(home_city=None),
    ))

    assert response is not None
    assert "starting from" in response.assistant_message
