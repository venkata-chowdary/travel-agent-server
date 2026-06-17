from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ai.schemas import TravelAgentStructuredResponse
from trips.models import Trip
from trips.schemas import TripCreate


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True)
    return value


def _trip_create_from_plan(plan: TravelAgentStructuredResponse) -> TripCreate:
    return TripCreate.model_validate(plan.model_dump(mode="json"))


async def create_trip(
    session: AsyncSession,
    user_id: str | UUID,
    payload: TripCreate | TravelAgentStructuredResponse,
) -> Trip:
    trip_data = _trip_create_from_plan(payload) if isinstance(payload, TravelAgentStructuredResponse) else payload

    trip = Trip(
        **({"id": trip_data.id} if trip_data.id is not None else {}),
        user_id=UUID(str(user_id)),
        destination=trip_data.destination,
        origin=trip_data.origin,
        start_date=trip_data.start_date,
        end_date=trip_data.end_date,
        days=trip_data.days,
        travelers=trip_data.travelers,
        status=trip_data.status,
        cover_emoji=trip_data.cover_emoji,
        summary=trip_data.summary,
        budget=_dump(trip_data.budget),
        itinerary=[_dump(day) for day in trip_data.itinerary],
        hotel_options=trip_data.hotel_options,
        flight_options=trip_data.flight_options,
        transport_options=[_dump(option) for option in trip_data.transport_options],
        daily_forecast=[_dump(day) for day in trip_data.daily_forecast],
        trip_risks=[_dump(risk) for risk in trip_data.trip_risks],
        verification_tips=trip_data.verification_tips,
        **({"created_at": trip_data.created_at} if trip_data.created_at is not None else {}),
    )
    session.add(trip)
    await session.commit()
    await session.refresh(trip)
    return trip


def _apply_trip_data(trip: Trip, trip_data: TripCreate) -> None:
    trip.destination = trip_data.destination
    trip.origin = trip_data.origin
    trip.start_date = trip_data.start_date
    trip.end_date = trip_data.end_date
    trip.days = trip_data.days
    trip.travelers = trip_data.travelers
    trip.status = trip_data.status
    trip.cover_emoji = trip_data.cover_emoji
    trip.summary = trip_data.summary
    trip.budget = _dump(trip_data.budget)
    trip.itinerary = [_dump(day) for day in trip_data.itinerary]
    trip.hotel_options = trip_data.hotel_options
    trip.flight_options = trip_data.flight_options
    trip.transport_options = [_dump(option) for option in trip_data.transport_options]
    trip.daily_forecast = [_dump(day) for day in trip_data.daily_forecast]
    trip.trip_risks = [_dump(risk) for risk in trip_data.trip_risks]
    trip.verification_tips = trip_data.verification_tips


async def update_trip_from_plan(
    session: AsyncSession,
    user_id: str | UUID,
    trip_id: str | UUID,
    payload: TripCreate | TravelAgentStructuredResponse,
) -> Trip | None:
    trip = await get_trip(session, user_id, trip_id)
    if trip is None:
        return None

    trip_data = _trip_create_from_plan(payload) if isinstance(payload, TravelAgentStructuredResponse) else payload
    _apply_trip_data(trip, trip_data)
    session.add(trip)
    await session.commit()
    await session.refresh(trip)
    return trip


async def list_trips(session: AsyncSession, user_id: str | UUID) -> list[Trip]:
    result = await session.execute(
        select(Trip)
        .where(Trip.user_id == UUID(str(user_id)))
        .order_by(Trip.created_at.desc())
    )
    return list(result.scalars().all())


async def get_trip(session: AsyncSession, user_id: str | UUID, trip_id: str | UUID) -> Trip | None:
    result = await session.execute(
        select(Trip).where(
            Trip.id == UUID(str(trip_id)),
            Trip.user_id == UUID(str(user_id)),
        )
    )
    return result.scalars().first()


async def delete_trip(session: AsyncSession, user_id: str | UUID, trip_id: str | UUID) -> bool:
    result = await session.execute(
        delete(Trip).where(
            Trip.id == UUID(str(trip_id)),
            Trip.user_id == UUID(str(user_id)),
        )
    )
    await session.commit()
    return bool(result.rowcount)


def trip_to_history(trip: Trip) -> dict[str, Any]:
    budget = trip.budget if isinstance(trip.budget, dict) else {}
    total = float(budget.get("total") or 0)
    per_day = total / trip.days if trip.days else total
    itinerary = trip.itinerary if isinstance(trip.itinerary, list) else []
    transports: set[str] = set()
    for day in itinerary:
        for item in day.get("items", []):
            if item.get("type") == "transport" and item.get("title"):
                transports.add(str(item["title"]))

    return {
        "destination": trip.destination,
        "duration_days": trip.days,
        "travel_style": trip.status,
        "accommodation": "saved trip plan",
        "transport": sorted(transports),
        "budget_per_day_inr": per_day,
        "travel_companions": f"{trip.travelers} traveler{'s' if trip.travelers != 1 else ''}",
        "pain_points": [],
        "overall_rating": 0,
    }
