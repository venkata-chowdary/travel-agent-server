import os

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/test")
os.environ.setdefault("JWT_SECRET_KEY", "x" * 32)

import pytest
from langchain_core.messages import AIMessage

import ai.agent as agent_module
from ai.agent import (
    ClarificationDecision,
    SupervisorDecision,
    _origin_from_state,
    _state_summary,
    clarifier_node,
    run_travel_agent,
    supervisor_node,
    transport_agent_node,
    weather_agent_node,
)
from ai.schemas import PreferenceContext, TravelAgentStructuredResponse, TravelPreferences
from ai.schemas.transport import TransportChoiceResponse
from ai.schemas.weather import WeatherForecastResponse
from ai.service import normalize_preference_payload


def base_state(message: str, **overrides):
    state = {
        "user_id": "user-1",
        "user_message": message,
        "messages": [],
        "next": "",
        "origin": None,
        "destination": None,
        "trip_duration_days": None,
        "trip_start_date": None,
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


def test_origin_from_state_uses_saved_preference_origin():
    state = base_state(
        "Plan a 3 day Goa trip",
        destination="Goa",
        trip_duration_days=3,
        preference_context=PreferenceContext(origin="Hyderabad"),
    )

    assert _origin_from_state(state) == "Hyderabad"


def test_origin_from_state_prefers_explicit_state_origin():
    state = base_state(
        "Plan a 3 day Goa trip",
        origin="Mumbai",
        preference_context=PreferenceContext(origin="Hyderabad"),
    )

    assert _origin_from_state(state) == "Mumbai"


def test_state_summary_exposes_preference_origin():
    summary = _state_summary(base_state(
        "Plan a 3 day Goa trip",
        destination="Goa",
        trip_duration_days=3,
        preference_context=PreferenceContext(origin="Hyderabad"),
    ))

    content = summary[0].content
    assert "preference_context: COLLECTED (origin: Hyderabad)" in content
    assert "- origin: Hyderabad" in content


def test_preference_models_normalize_legacy_home_city():
    prefs = TravelPreferences.model_validate({"home_city": "Hyderabad"})
    ctx = PreferenceContext.model_validate({"home_city": "Hyderabad"})

    assert prefs.origin == "Hyderabad"
    assert ctx.origin == "Hyderabad"


def test_preference_payload_normalizer_keeps_existing_origin():
    assert normalize_preference_payload({"home_city": "Hyderabad"})["origin"] == "Hyderabad"
    assert normalize_preference_payload({"origin": "Mumbai", "home_city": "Hyderabad"})["origin"] == "Mumbai"


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


class FakeStructuredLLM:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = []
        self.schema = None

    def with_structured_output(self, schema, method=None):
        self.schema = schema
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        output = self.outputs.pop(0)
        if isinstance(output, self.schema):
            return output
        return self.schema.model_validate(output)


def resolved_workflow(**overrides):
    statuses = {
        "preferences": "succeeded",
        "clarification": "succeeded",
        "weather": "succeeded",
        "transport": "succeeded",
    }
    statuses.update(overrides)
    return statuses


@pytest.mark.asyncio
async def test_transport_agent_marks_success_status():
    result = await transport_agent_node(base_state(
        "Plan Goa",
        origin="Hyderabad",
        destination="GOA",
        trip_duration_days=2,
        trip_start_date="2026-06-20",
    ))

    assert result["workflow_statuses"]["transport"] == "succeeded"
    assert result["transport_choice"].outbound_options


@pytest.mark.asyncio
async def test_transport_agent_marks_empty_status(monkeypatch):
    def fake_choice(**kwargs):
        return TransportChoiceResponse(
            origin=kwargs["origin"],
            destination=kwargs["destination"],
            start_date=kwargs["start_date"],
            days=kwargs["days"],
            travelers=kwargs["travelers"],
            summary="No transport options found.",
        )

    monkeypatch.setattr(agent_module, "build_transport_choice", fake_choice)

    result = await transport_agent_node(base_state(
        "Plan Goa",
        origin="Hyderabad",
        destination="Goa",
        trip_duration_days=2,
        trip_start_date="2026-06-20",
    ))

    assert result["workflow_statuses"]["transport"] == "empty"
    assert result["transport_choice"].outbound_options == []


@pytest.mark.asyncio
async def test_weather_agent_failure_is_non_success_status(monkeypatch):
    class FailingWeatherAgent:
        async def ainvoke(self, payload):
            raise RuntimeError("weather unavailable")

    monkeypatch.setattr(agent_module, "build_weather_executor", lambda llm: FailingWeatherAgent())

    result = await weather_agent_node(base_state(
        "Plan Goa",
        destination="Goa",
        trip_duration_days=2,
    ))

    assert result["workflow_statuses"]["weather"] == "failed"
    assert result["weather_forecast"].daily_forecast == []


@pytest.mark.asyncio
async def test_clarifier_question_marks_waiting_for_user(monkeypatch):
    fake_llm = FakeStructuredLLM(ClarificationDecision(
        needs_clarification=True,
        questions=["Where are you starting from?"],
        assistant_message="Where are you starting from?",
    ))
    monkeypatch.setattr(agent_module, "_llm", fake_llm)

    result = await clarifier_node(base_state("Plan a 2 day Goa trip"))

    assert result["workflow_statuses"]["clarification"] == "waiting_for_user"
    assert result["clarification_response"].response_type == "clarification"


@pytest.mark.asyncio
async def test_supervisor_retry_empty_transport_reroutes_and_clears_stale_choice(monkeypatch):
    empty_choice = TransportChoiceResponse(
        origin="Hyderabad",
        destination="Goa",
        start_date="2026-06-20",
        days=2,
        travelers=1,
        summary="No transport options found.",
    )
    fake_llm = FakeStructuredLLM(SupervisorDecision(
        intent="retry_step",
        target_step="transport",
        next="transport_agent",
        origin="Hyderabad",
        destination="Goa",
        trip_duration_days=2,
    ))
    monkeypatch.setattr(agent_module, "_llm", fake_llm)

    result = await supervisor_node(base_state(
        "check again",
        origin="Hyderabad",
        destination="Goa",
        trip_duration_days=2,
        transport_choice=empty_choice,
        workflow_statuses=resolved_workflow(transport="empty"),
    ))

    assert result["next"] == "transport_agent"
    assert result["workflow_statuses"]["transport"] == "not_started"
    assert result["transport_choice"] is None
    assert result["selected_transport_options"] is None


@pytest.mark.asyncio
async def test_supervisor_retry_failed_weather_reroutes_and_clears_stale_forecast(monkeypatch):
    fake_llm = FakeStructuredLLM(SupervisorDecision(
        intent="retry_step",
        target_step="weather",
        next="weather_agent",
        origin="Hyderabad",
        destination="Goa",
        trip_duration_days=2,
    ))
    monkeypatch.setattr(agent_module, "_llm", fake_llm)

    result = await supervisor_node(base_state(
        "try the weather again",
        origin="Hyderabad",
        destination="Goa",
        trip_duration_days=2,
        weather_forecast=WeatherForecastResponse(
            destination="Goa",
            summary="Weather unavailable.",
            daily_forecast=[],
            trip_risks=[],
            requires_replanning=False,
        ),
        workflow_statuses=resolved_workflow(weather="failed"),
    ))

    assert result["next"] == "weather_agent"
    assert result["workflow_statuses"]["weather"] == "not_started"
    assert result["weather_forecast"] is None


@pytest.mark.asyncio
async def test_supervisor_proceed_without_empty_transport_marks_skipped(monkeypatch):
    empty_choice = TransportChoiceResponse(
        origin="Hyderabad",
        destination="Goa",
        start_date="2026-06-20",
        days=2,
        travelers=1,
        summary="No transport options found.",
    )
    fake_llm = FakeStructuredLLM(SupervisorDecision(
        intent="proceed_without_step",
        target_step="transport",
        next="planner",
        origin="Hyderabad",
        destination="Goa",
        trip_duration_days=2,
    ))
    monkeypatch.setattr(agent_module, "_llm", fake_llm)

    result = await supervisor_node(base_state(
        "continue without transport",
        origin="Hyderabad",
        destination="Goa",
        trip_duration_days=2,
        transport_choice=empty_choice,
        workflow_statuses=resolved_workflow(transport="empty"),
    ))

    assert result["next"] == "planner"
    assert result["workflow_statuses"]["transport"] == "skipped_by_user"
    assert result["transport_choice"] is None


@pytest.mark.asyncio
async def test_supervisor_reasks_ai_after_incompatible_planner_decision(monkeypatch):
    fake_llm = FakeStructuredLLM(
        SupervisorDecision(
            intent="start_or_continue",
            target_step="planner",
            next="planner",
            destination="Goa",
        ),
        SupervisorDecision(
            intent="ask_clarification",
            target_step="clarification",
            next="clarifier",
            destination="Goa",
        ),
    )
    monkeypatch.setattr(agent_module, "_llm", fake_llm)

    result = await supervisor_node(base_state(
        "make the plan",
        destination="Goa",
        workflow_statuses=resolved_workflow(),
    ))

    assert result["next"] == "clarifier"
    assert len(fake_llm.calls) == 2


@pytest.mark.asyncio
async def test_supervisor_all_resolved_allows_planner(monkeypatch):
    fake_llm = FakeStructuredLLM(SupervisorDecision(
        intent="start_or_continue",
        target_step="planner",
        next="planner",
        origin="Hyderabad",
        destination="Goa",
        trip_duration_days=2,
    ))
    monkeypatch.setattr(agent_module, "_llm", fake_llm)

    result = await supervisor_node(base_state(
        "make the plan",
        origin="Hyderabad",
        destination="Goa",
        trip_duration_days=2,
        workflow_statuses=resolved_workflow(),
    ))

    assert result["next"] == "planner"
