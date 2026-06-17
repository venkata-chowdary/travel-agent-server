import os
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import UUID, uuid4

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/test")
os.environ.setdefault("JWT_SECRET_KEY", "x" * 32)

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai.schemas import TravelAgentChatResponse, TravelAgentStructuredResponse
from auth.dependencies import get_current_user
from db import get_db_session
from routes.agent import router as agent_router
import routes.agent as agent_routes
from trips.routes import router as trips_router
import trips.routes as trip_routes


USER_ID = UUID("11111111-1111-1111-1111-111111111111")
OTHER_ID = UUID("22222222-2222-2222-2222-222222222222")
TEST_SESSION_ID = "test-session-00000000"


def make_trip(**overrides):
    now = datetime.now(timezone.utc)
    data = {
        "id": uuid4(),
        "user_id": USER_ID,
        "destination": "Goa",
        "origin": None,
        "start_date": None,
        "end_date": None,
        "days": 3,
        "travelers": 2,
        "status": "planning",
        "cover_emoji": "beach",
        "summary": "Beach plan",
        "budget": {
            "flights": 100,
            "stay": 200,
            "activities": 50,
            "food": 40,
            "total": 390,
            "currency": "INR",
        },
        "itinerary": [
            {
                "day": 1,
                "title": "Arrival",
                "items": [
                    {
                        "time": "10:00",
                        "title": "Check in",
                        "description": "Drop bags",
                        "type": "stay",
                    }
                ],
            }
        ],
        "hotel_options": [],
        "flight_options": [],
        "transport_options": [],
        "daily_forecast": [],
        "trip_risks": [],
        "verification_tips": [],
        "created_at": now,
        "updated_at": now,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def make_app(authenticated: bool = True) -> FastAPI:
    app = FastAPI()
    app.include_router(trips_router)
    app.include_router(agent_router)

    async def fake_session():
        yield SimpleNamespace()

    app.dependency_overrides[get_db_session] = fake_session
    if authenticated:
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=USER_ID)
    return app


def test_trip_endpoints_require_authentication():
    app = make_app(authenticated=False)
    client = TestClient(app)

    assert client.get("/api/trips").status_code == 401
    assert client.get(f"/api/trips/{uuid4()}").status_code == 401
    assert client.delete(f"/api/trips/{uuid4()}").status_code == 401


def test_chat_history_returns_current_user_session_messages(monkeypatch):
    now = datetime.now(timezone.utc)
    rows = [
        SimpleNamespace(id=uuid4(), role="user", content="Plan Goa", created_at=now, payload={}),
        SimpleNamespace(
            id=uuid4(),
            role="assistant",
            content="I found options.",
            created_at=now,
            payload={
                "response_type": "transport_choice",
                "transport_choice": {"origin": "Hyderabad", "destination": "Goa"},
            },
        ),
    ]

    async def fake_load_chat_history(session, session_id, user_id=None):
        assert session_id == TEST_SESSION_ID
        assert user_id == USER_ID
        return rows

    monkeypatch.setattr(agent_routes, "load_chat_history", fake_load_chat_history)

    client = TestClient(make_app())
    response = client.get(f"/api/agent/history/{TEST_SESSION_ID}")

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": str(rows[0].id),
            "role": "user",
            "content": "Plan Goa",
            "created_at": now.isoformat(),
            "payload": {},
        },
        {
            "id": str(rows[1].id),
            "role": "assistant",
            "content": "I found options.",
            "created_at": now.isoformat(),
            "payload": {
                "response_type": "transport_choice",
                "transport_choice": {"origin": "Hyderabad", "destination": "Goa"},
            },
        },
    ]


def test_list_create_detail_and_delete_trips(monkeypatch):
    trip = make_trip()

    async def fake_list_trips(session, user_id):
        assert user_id == USER_ID
        return [trip]

    async def fake_create_trip(session, user_id, payload):
        assert user_id == USER_ID
        assert payload.destination == "Goa"
        return make_trip(id=payload.id or trip.id)

    async def fake_get_trip(session, user_id, trip_id):
        assert user_id == USER_ID
        return trip if trip_id == trip.id else None

    async def fake_delete_trip(session, user_id, trip_id):
        assert user_id == USER_ID
        return trip_id == trip.id

    monkeypatch.setattr(trip_routes, "list_trips", fake_list_trips)
    monkeypatch.setattr(trip_routes, "create_trip", fake_create_trip)
    monkeypatch.setattr(trip_routes, "get_trip", fake_get_trip)
    monkeypatch.setattr(trip_routes, "delete_trip", fake_delete_trip)

    client = TestClient(make_app())
    payload = {
        "id": str(trip.id),
        "destination": "Goa",
        "days": 3,
        "travelers": 2,
        "status": "planning",
        "cover_emoji": "beach",
        "summary": "Beach plan",
        "budget": trip.budget,
        "itinerary": trip.itinerary,
        "hotel_options": [],
        "flight_options": [],
        "transport_options": [],
    }

    listed = client.get("/api/trips")
    assert listed.status_code == 200
    assert listed.json()[0]["id"] == str(trip.id)

    created = client.post("/api/trips", json=payload)
    assert created.status_code == 201
    assert created.json()["destination"] == "Goa"

    detail = client.get(f"/api/trips/{trip.id}")
    assert detail.status_code == 200
    assert detail.json()["user_id"] == str(USER_ID)

    missing = client.get(f"/api/trips/{OTHER_ID}")
    assert missing.status_code == 404

    deleted = client.delete(f"/api/trips/{trip.id}")
    assert deleted.status_code == 204

    missing_delete = client.delete(f"/api/trips/{OTHER_ID}")
    assert missing_delete.status_code == 404


def test_agent_chat_persists_generated_plan(monkeypatch):
    saved = {}
    plan = TravelAgentStructuredResponse(
        id=str(uuid4()),
        destination="Tokyo",
        days=5,
        travelers=1,
        summary="A food and city plan",
        itinerary=[
            {
                "day": 1,
                "title": "Arrival",
                "items": [
                    {
                        "time": "18:00",
                        "title": "Ramen dinner",
                        "description": "First dinner in Shinjuku",
                        "type": "meal",
                    }
                ],
            }
        ],
        budget={
            "flights": 1000,
            "stay": 700,
            "activities": 200,
            "food": 300,
            "total": 2200,
            "currency": "INR",
        },
        transport_options=[
            {
                "id": "outbound_flt_001",
                "mode": "flight",
                "leg": "outbound",
                "provider": "IndiGo",
                "from": "HYD",
                "to": "GOI",
                "depart": "08:00",
                "arrive": "09:30",
                "duration": "1h 30m",
                "price": 4200,
                "available_seats": 24,
                "rating": 4.3,
                "details": {"flight_number": "6E-245"},
            }
        ],
    )

    async def fake_run_travel_agent(user_id, message, session_id, history=None, transport_selection=None):
        assert user_id == USER_ID
        assert message == "Plan Tokyo"
        assert session_id == TEST_SESSION_ID
        assert transport_selection is None
        return TravelAgentChatResponse(
            response_type="trip_plan",
            assistant_message="Here's a Tokyo plan.",
            trip_plan=plan,
        )

    async def fake_load_chat_history(session, session_id, user_id=None):
        assert user_id == USER_ID
        return []

    async def fake_save_chat_turn(
        session,
        session_id,
        user_id,
        user_content,
        assistant_content,
        user_payload=None,
        assistant_payload=None,
    ):
        assert user_payload == {}
        assert assistant_payload["response_type"] == "trip_plan"
        assert assistant_payload["trip_plan"]["destination"] == "Tokyo"

    async def fake_create_trip(session, user_id, payload):
        saved["user_id"] = user_id
        saved["payload"] = payload
        return make_trip(id=UUID(payload.id), destination=payload.destination)

    monkeypatch.setattr(agent_routes, "run_travel_agent", fake_run_travel_agent)
    monkeypatch.setattr(agent_routes, "load_chat_history", fake_load_chat_history)
    monkeypatch.setattr(agent_routes, "save_chat_turn", fake_save_chat_turn)
    monkeypatch.setattr(agent_routes, "create_trip", fake_create_trip)

    client = TestClient(make_app())
    response = client.post("/api/agent/chat", json={"message": "Plan Tokyo", "session_id": TEST_SESSION_ID})

    assert response.status_code == 200
    assert response.json()["response_type"] == "trip_plan"
    assert response.json()["trip_plan"]["destination"] == "Tokyo"
    assert response.json()["trip_plan"]["transport_options"][0]["id"] == "outbound_flt_001"
    assert saved["user_id"] == USER_ID
    assert saved["payload"] == plan


def test_agent_chat_with_target_trip_updates_existing_trip(monkeypatch):
    trip_id = uuid4()
    saved = {}
    existing_trip = make_trip(id=trip_id, destination="Goa")
    plan = TravelAgentStructuredResponse(
        id=str(uuid4()),
        destination="Goa",
        days=4,
        travelers=2,
        summary="Updated Goa plan",
        itinerary=[],
        budget={
            "flights": 100,
            "stay": 200,
            "activities": 50,
            "food": 40,
            "total": 390,
            "currency": "INR",
        },
    )

    async def fake_get_trip(session, user_id, requested_trip_id):
        assert user_id == USER_ID
        assert requested_trip_id == trip_id
        return existing_trip

    async def fake_run_travel_agent(user_id, message, session_id, history=None, transport_selection=None):
        assert user_id == USER_ID
        assert message == "Make it four days"
        assert session_id == TEST_SESSION_ID
        assert history
        assert "editing an existing saved trip" in history[0].content
        return TravelAgentChatResponse(
            response_type="trip_plan",
            assistant_message="Here's an updated Goa plan.",
            trip_plan=plan,
        )

    async def fake_load_chat_history(session, session_id, user_id=None):
        assert user_id == USER_ID
        return []

    async def fake_save_chat_turn(
        session,
        session_id,
        user_id,
        user_content,
        assistant_content,
        user_payload=None,
        assistant_payload=None,
    ):
        assert user_payload == {"target_trip_id": str(trip_id)}
        assert assistant_payload["response_type"] == "trip_plan"
        assert assistant_payload["trip_plan"]["id"] == str(trip_id)

    async def fake_update_trip_from_plan(session, user_id, requested_trip_id, payload):
        saved["updated"] = True
        saved["trip_id"] = requested_trip_id
        saved["payload"] = payload
        return make_trip(id=requested_trip_id, destination=payload.destination)

    async def fail_create_trip(session, user_id, payload):
        raise AssertionError("editing an existing trip must not create a duplicate")

    monkeypatch.setattr(agent_routes, "get_trip", fake_get_trip)
    monkeypatch.setattr(agent_routes, "run_travel_agent", fake_run_travel_agent)
    monkeypatch.setattr(agent_routes, "load_chat_history", fake_load_chat_history)
    monkeypatch.setattr(agent_routes, "save_chat_turn", fake_save_chat_turn)
    monkeypatch.setattr(agent_routes, "update_trip_from_plan", fake_update_trip_from_plan)
    monkeypatch.setattr(agent_routes, "create_trip", fail_create_trip)

    client = TestClient(make_app())
    response = client.post(
        "/api/agent/chat",
        json={
            "message": "Make it four days",
            "session_id": TEST_SESSION_ID,
            "target_trip_id": str(trip_id),
        },
    )

    assert response.status_code == 200
    assert response.json()["trip_plan"]["id"] == str(trip_id)
    assert saved["updated"] is True
    assert saved["trip_id"] == trip_id
    assert saved["payload"].id == str(trip_id)


def test_agent_chat_clarification_does_not_persist(monkeypatch):
    async def fake_run_travel_agent(user_id, message, session_id, history=None, transport_selection=None):
        assert user_id == USER_ID
        assert message == "Plan a trip"
        assert session_id == TEST_SESSION_ID
        assert transport_selection is None
        return TravelAgentChatResponse(
            response_type="clarification",
            assistant_message="Nice, I can plan that. Where do you want to go?",
            questions=["Where do you want to go?"],
        )

    async def fake_load_chat_history(session, session_id, user_id=None):
        assert user_id == USER_ID
        return []

    async def fake_save_chat_turn(
        session,
        session_id,
        user_id,
        user_content,
        assistant_content,
        user_payload=None,
        assistant_payload=None,
    ):
        assert user_payload == {}
        assert assistant_payload["response_type"] == "clarification"

    async def fail_create_trip(session, user_id, payload):
        raise AssertionError("clarification responses must not be persisted")

    monkeypatch.setattr(agent_routes, "run_travel_agent", fake_run_travel_agent)
    monkeypatch.setattr(agent_routes, "load_chat_history", fake_load_chat_history)
    monkeypatch.setattr(agent_routes, "save_chat_turn", fake_save_chat_turn)
    monkeypatch.setattr(agent_routes, "create_trip", fail_create_trip)

    client = TestClient(make_app())
    response = client.post("/api/agent/chat", json={"message": "Plan a trip", "session_id": TEST_SESSION_ID})

    assert response.status_code == 200
    assert response.json()["response_type"] == "clarification"
    assert response.json()["trip_plan"] is None
    assert response.json()["questions"] == ["Where do you want to go?"]
