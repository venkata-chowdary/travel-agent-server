from types import SimpleNamespace
from uuid import uuid4

from ai.chat_service import _trip_context_message


def test_trip_context_message_includes_saved_grounding_context() -> None:
    trip = SimpleNamespace(
        id=uuid4(),
        destination="Goa",
        origin="Bengaluru",
        start_date="2026-07-10",
        end_date="2026-07-12",
        days=3,
        travelers=2,
        summary="A relaxed beach break.",
        budget={"flights": 12000, "stay": 18000, "activities": 4000, "food": 3000, "total": 37000, "currency": "INR"},
        itinerary=[
            {
                "day": 1,
                "date": "2026-07-10",
                "title": "Arrival",
                "items": [
                    {"time": "16:00", "title": "Check-in", "description": "Settle in", "type": "stay"},
                    {"time": "19:00", "title": "Gunpowder", "description": "Goan dinner", "type": "meal"},
                ],
            },
            {
                "day": 2,
                "date": "2026-07-11",
                "title": "Beach day",
                "items": [
                    {"time": "10:00", "title": "Calangute Beach", "description": "Morning swim", "type": "activity"},
                ],
            },
        ],
        transport_options=[
            {
                "id": "flight_1",
                "mode": "flight",
                "leg": "outbound",
                "provider": "IndiGo",
                "from": "Bengaluru",
                "to": "Goa",
                "price": 6000,
            }
        ],
        hotel_options=[
            {
                "id": "hotel_1",
                "name": "Casa Shoreline",
                "area": "Candolim",
                "hotel_type": "mid_range",
                "price_per_night": 9000,
                "total_price": 18000,
                "rating": 4.4,
                "amenities": ["wifi", "pool"],
                "breakfast_included": True,
                "refundable": True,
            }
        ],
        daily_forecast=[
            {
                "day": 2,
                "date": "2026-07-11",
                "condition": "Heavy rain",
                "temperature": "27C",
                "rain_probability": 90,
                "risk_level": "high",
            }
        ],
        trip_risks=[
            {
                "day": 2,
                "risk_type": "rain",
                "severity": "high",
                "recommendation": "Prefer indoor backup activities.",
            }
        ],
        verification_tips=["Recheck beach activity availability during monsoon."],
    )

    content = _trip_context_message(trip).content

    assert "Selected hotel: Casa Shoreline (mid_range) in Candolim" in content
    assert "Weather context from saved trip:" in content
    assert "Day 2 [high] rain: Prefer indoor backup activities." in content
    assert "Activity and restaurant context from saved itinerary:" in content
    assert "Calangute Beach - Morning swim" in content
    assert "Gunpowder - Goan dinner" in content
    assert "Recheck beach activity availability during monsoon." in content
