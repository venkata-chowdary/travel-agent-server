import pytest
from langchain_core.messages import AIMessage

import ai.agent as agent_module


class DummyGraph:
    def __init__(self, content: str = "Draft travel response"):
        self._content = content

    def invoke(self, *_args, **_kwargs):
        return {"messages": [AIMessage(content=self._content)]}


class DummyStructuredLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def invoke(self, _messages):
        next_value = self.responses.pop(0)
        if isinstance(next_value, Exception):
            raise next_value
        return next_value


class DummyModel:
    def __init__(self, structured_llm):
        self.structured_llm = structured_llm

    def with_structured_output(self, _schema):
        return self.structured_llm


async def _mock_preferences(_user_id):
    return {"budgetMin": 25000, "budgetMax": 60000, "travelStyle": "balanced"}


@pytest.mark.asyncio
async def test_run_travel_agent_returns_structured_payload(monkeypatch):
    expected = agent_module.TravelAgentStructuredResponse(
        destination="Goa",
        days=4,
        travelers=2,
        trip_overview="A balanced 4-day Goa itinerary for two.",
        itinerary=[],
        budget=agent_module.BudgetBreakdown(
            flights=12000,
            stay=18000,
            activities=7000,
            food=5000,
            total=42000,
            currency="INR",
        ),
        weather=agent_module.WeatherNotes(summary="Mostly sunny with brief evening showers.", confidence="medium"),
    )

    monkeypatch.setattr(agent_module, "fetch_user_preferences", _mock_preferences)
    monkeypatch.setattr(agent_module, "compiled_graph", DummyGraph())
    monkeypatch.setattr(agent_module, "llm", DummyModel(DummyStructuredLLM([expected])))

    result = await agent_module.run_travel_agent("00000000-0000-0000-0000-000000000001", "Plan Goa for 4 days")
    assert result.destination == "Goa"
    assert result.budget.total == 42000


@pytest.mark.asyncio
async def test_run_travel_agent_retries_once_before_success(monkeypatch):
    repaired = agent_module.TravelAgentStructuredResponse(
        destination="Tokyo",
        days=5,
        travelers=1,
        trip_overview="A practical 5-day Tokyo plan.",
        itinerary=[],
        budget=agent_module.BudgetBreakdown(
            flights=45000,
            stay=40000,
            activities=15000,
            food=10000,
            total=110000,
            currency="JPY",
        ),
        weather=agent_module.WeatherNotes(summary="Cool and clear conditions expected.", confidence="high"),
    )

    monkeypatch.setattr(agent_module, "fetch_user_preferences", _mock_preferences)
    monkeypatch.setattr(agent_module, "compiled_graph", DummyGraph())
    monkeypatch.setattr(
        agent_module,
        "llm",
        DummyModel(DummyStructuredLLM([Exception("schema mismatch"), repaired])),
    )

    result = await agent_module.run_travel_agent("00000000-0000-0000-0000-000000000001", "Tokyo plan")
    assert result.destination == "Tokyo"
    assert result.days == 5


@pytest.mark.asyncio
async def test_run_travel_agent_returns_fallback_after_double_failure(monkeypatch):
    monkeypatch.setattr(agent_module, "fetch_user_preferences", _mock_preferences)
    monkeypatch.setattr(agent_module, "compiled_graph", DummyGraph("Need more details from user."))
    monkeypatch.setattr(
        agent_module,
        "llm",
        DummyModel(DummyStructuredLLM([Exception("first fail"), Exception("second fail")])),
    )

    result = await agent_module.run_travel_agent("00000000-0000-0000-0000-000000000001", "Help me plan")
    assert len(result.verification_tips) >= 1
    assert result.trip_overview
