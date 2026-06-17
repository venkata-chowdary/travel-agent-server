import os

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/test")
os.environ.setdefault("JWT_SECRET_KEY", "x" * 32)

import pytest
from langchain_core.messages import AIMessage

import ai.agent as agent_module
from ai.agent import _clarification_response, run_travel_agent
from ai.schemas import PreferenceContext, TravelAgentStructuredResponse


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


@pytest.mark.asyncio
async def test_run_travel_agent_uses_checkpoint_thread_without_empty_preferences(monkeypatch):
    captured = {}

    class FakeAgent:
        async def ainvoke(self, input_state, config=None):
            captured["input_state"] = input_state
            captured["config"] = config
            return {
                "structured_response": TravelAgentStructuredResponse(
                    destination="Chennai",
                    days=2,
                    travelers=1,
                    summary="A short Chennai plan",
                    itinerary=[
                        {
                            "day": 1,
                            "title": "Arrival",
                            "items": [
                                {
                                    "time": "10:00",
                                    "title": "Arrive",
                                    "description": "Reach Chennai",
                                    "type": "transport",
                                }
                            ],
                        }
                    ],
                    budget={
                        "flights": 100,
                        "stay": 200,
                        "activities": 100,
                        "food": 100,
                        "total": 500,
                        "currency": "INR",
                    },
                )
            }

    monkeypatch.setattr(agent_module, "agent", FakeAgent())

    response = await run_travel_agent(
        "user-1",
        "tomorrow",
        "session-123",
        history=[AIMessage(content="Where will you be starting from?")],
    )

    assert response.response_type == "trip_plan"
    assert captured["config"] == {"configurable": {"thread_id": "session-123"}}
    assert "preference_context" not in captured["input_state"]
    assert "origin" not in captured["input_state"]
    assert "destination" not in captured["input_state"]
    assert captured["input_state"]["clarification_checked"] is False
