from __future__ import annotations

import uuid as _uuid
from typing import Any
from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from ai.schemas import TravelAgentStructuredResponse
from ai.schemas.transport import TransportChoiceResponse, TransportOption
from trips.models import Trip, TripTransportOption
from trips.schemas import TripCreate


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True)
    return value


async def create_trip(
    session: AsyncSession,
    user_id: str | UUID,
    payload: TripCreate | TravelAgentStructuredResponse,
    session_id: str | None = None,
) -> Trip:
    trip_data = TripCreate.model_validate(payload.model_dump(mode="json")) if isinstance(payload, TravelAgentStructuredResponse) else payload

    trip = Trip(
        **({"id": trip_data.id} if trip_data.id is not None else {}),
        user_id=UUID(str(user_id)),
        session_id=session_id,
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

    trip_data = TripCreate.model_validate(payload.model_dump(mode="json")) if isinstance(payload, TravelAgentStructuredResponse) else payload
    _apply_trip_data(trip, trip_data)
    session.add(trip)
    await session.commit()
    await session.refresh(trip)
    return trip


async def list_trips(
    session: AsyncSession,
    user_id: str | UUID,
    limit: int = 50,
    offset: int = 0,
) -> list[Trip]:
    result = await session.execute(
        select(Trip)
        .where(Trip.user_id == UUID(str(user_id)))
        .order_by(Trip.created_at.desc())
        .limit(limit)
        .offset(offset)
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


async def update_trip_status(
    session: AsyncSession,
    user_id: str | UUID,
    trip_id: str | UUID,
    new_status: str,
) -> Trip | None:
    trip = await get_trip(session, user_id, trip_id)
    if trip is None:
        return None
    trip.status = new_status
    session.add(trip)
    await session.commit()
    await session.refresh(trip)
    return trip


async def delete_trip(session: AsyncSession, user_id: str | UUID, trip_id: str | UUID) -> bool:
    result = await session.execute(
        delete(Trip).where(
            Trip.id == UUID(str(trip_id)),
            Trip.user_id == UUID(str(user_id)),
        )
    )
    await session.commit()
    return bool(result.rowcount)


async def create_draft_trip(
    session: AsyncSession,
    user_id: str | UUID,
    session_id: str,
    transport_choice: TransportChoiceResponse,
) -> Trip:
    """Create a minimal trip stub as soon as the clarifier passes and transport options are ready.

    Uses INSERT … ON CONFLICT DO UPDATE so that concurrent requests (double-tap, retry)
    targeting the same (user_id, session_id) draft slot converge on a single row rather
    than creating duplicates.
    """
    stmt = (
        pg_insert(Trip)
        .values(
            id=_uuid.uuid4(),
            user_id=UUID(str(user_id)),
            session_id=session_id,
            destination=transport_choice.destination,
            origin=transport_choice.origin,
            start_date=transport_choice.start_date,
            end_date=transport_choice.end_date,
            days=transport_choice.days,
            travelers=transport_choice.travelers,
            status="draft",
            summary=f"Trip to {transport_choice.destination}",
            budget={"flights": 0, "stay": 0, "activities": 0, "food": 0, "total": 0, "currency": "₹"},
        )
        .on_conflict_do_update(
            index_elements=["user_id", "session_id"],
            index_where=Trip.status == "draft",
            set_=dict(
                destination=transport_choice.destination,
                origin=transport_choice.origin,
                start_date=transport_choice.start_date,
                end_date=transport_choice.end_date,
                days=transport_choice.days,
                travelers=transport_choice.travelers,
                summary=f"Trip to {transport_choice.destination}",
            ),
        )
    )
    await session.execute(stmt)
    await session.commit()

    result = await session.execute(
        select(Trip).where(
            Trip.user_id == UUID(str(user_id)),
            Trip.session_id == session_id,
            Trip.status == "draft",
        )
    )
    return result.scalars().first()


async def get_draft_trip_by_session(
    session: AsyncSession,
    user_id: str | UUID,
    session_id: str,
) -> Trip | None:
    result = await session.execute(
        select(Trip).where(
            Trip.user_id == UUID(str(user_id)),
            Trip.session_id == session_id,
            Trip.status == "draft",
        )
    )
    return result.scalars().first()


async def get_trip_by_session(
    session: AsyncSession,
    user_id: str | UUID,
    session_id: str,
) -> Trip | None:
    """Find any trip (draft or finalized) created from a given session.

    Used to prevent duplicate trip creation when the planner retries in the same session.
    """
    result = await session.execute(
        select(Trip).where(
            Trip.user_id == UUID(str(user_id)),
            Trip.session_id == session_id,
        ).order_by(Trip.created_at.desc())
    )
    return result.scalars().first()


async def save_transport_options(
    session: AsyncSession,
    session_id: str,
    trip_id: UUID,
    transport_choice: TransportChoiceResponse,
) -> None:
    """Persist all available transport options linked to the draft trip.

    trip_id is always set here (draft trip is created first), so trip_transport_options
    never has a NULL trip_id. ON CONFLICT DO NOTHING makes retries safe.
    """
    recommended_ids = {
        transport_choice.recommended_outbound_id,
        transport_choice.recommended_return_id,
    } - {None}

    rows = [
        {
            "session_id": session_id,
            "trip_id": trip_id,
            "option_id": opt.id,
            "mode": opt.mode,
            "leg": opt.leg,
            "provider": opt.provider,
            "from_city": opt.from_,
            "to_city": opt.to,
            "depart": opt.depart,
            "arrive": opt.arrive,
            "duration": opt.duration,
            "price": opt.price,
            "available_seats": opt.available_seats,
            "rating": opt.rating,
            "details": opt.details,
            "status": "available",
            "is_recommended": opt.id in recommended_ids,
        }
        for opt in transport_choice.outbound_options + transport_choice.return_options
    ]
    if not rows:
        return

    stmt = pg_insert(TripTransportOption).values(rows).on_conflict_do_nothing(
        index_elements=["session_id", "option_id"]
    )
    await session.execute(stmt)
    await session.commit()


async def link_transport_options_to_trip(
    session: AsyncSession,
    trip_id: UUID,
    selected_options: list[TransportOption],
) -> None:
    """Finalise transport option statuses on the trip after the planner runs.

    Options are already linked to trip_id (set when the draft trip was created).
    This call just stamps the final status on each option and updates trip.transport_status.

    - selected options  → 'selected'
    - remaining options → 'available' (visible as alternatives in the trip detail view)
    - if user skipped   → all options → 'skipped'
    - no options at all → trip.transport_status stays 'not_searched'
    """
    count_row = await session.execute(
        select(func.count()).select_from(TripTransportOption)
        .where(TripTransportOption.trip_id == trip_id)
    )
    if (count_row.scalar() or 0) == 0:
        return

    selected_ids = {opt.id for opt in selected_options}

    if not selected_ids:
        await session.execute(
            update(TripTransportOption)
            .where(TripTransportOption.trip_id == trip_id)
            .values(status="skipped")
        )
        transport_status = "skipped"
    else:
        await session.execute(
            update(TripTransportOption)
            .where(
                TripTransportOption.trip_id == trip_id,
                TripTransportOption.option_id.in_(selected_ids),
            )
            .values(status="selected")
        )
        transport_status = "selected"

    await session.execute(
        update(Trip).where(Trip.id == trip_id).values(transport_status=transport_status)
    )
    await session.commit()


async def expire_session_transport_options(session: AsyncSession, session_id: str) -> None:
    """Mark all still-available options for a session as expired.

    Call when a session is abandoned or a new trip flow starts for the same session_id.
    """
    await session.execute(
        update(TripTransportOption)
        .where(
            TripTransportOption.session_id == session_id,
            TripTransportOption.status == "available",
        )
        .values(status="expired")
    )
    await session.commit()


async def get_trip_transport_options(
    session: AsyncSession,
    trip_id: UUID,
) -> list[TripTransportOption]:
    result = await session.execute(
        select(TripTransportOption)
        .where(TripTransportOption.trip_id == trip_id)
        .order_by(TripTransportOption.leg, TripTransportOption.mode, TripTransportOption.price)
    )
    return list(result.scalars().all())


async def get_session_transport_options(
    session: AsyncSession,
    session_id: str,
    user_id: str | UUID,
) -> list[TripTransportOption]:
    """Return available (not yet selected) options for a session — used for auto-resume."""
    result = await session.execute(
        select(TripTransportOption)
        .join(Trip, TripTransportOption.trip_id == Trip.id)
        .where(
            TripTransportOption.session_id == session_id,
            TripTransportOption.status == "available",
            Trip.user_id == UUID(str(user_id)),
        )
        .order_by(TripTransportOption.leg, TripTransportOption.mode, TripTransportOption.price)
    )
    return list(result.scalars().all())


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
        "travel_style": "",
        "status": trip.status,
        "accommodation": "saved trip plan",
        "transport": sorted(transports),
        "budget_per_day_inr": per_day,
        "travel_companions": f"{trip.travelers} traveler{'s' if trip.travelers != 1 else ''}",
        "pain_points": [],
        "overall_rating": 0,
    }
